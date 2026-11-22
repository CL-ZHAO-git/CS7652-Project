"""CRSP daily stock data from WRDS, prepared for the forecasting panel.

Restricted data and credentials stay out of the agent runtime and version control:

1. ``fetch_raw`` (needs WRDS access and the optional ``wrds`` package) downloads a
   raw extract for one experiment window into a git-ignored directory.
2. ``build_panel`` (standard library only) turns that extract into the panel JSON
   read by ``[data] source = "json"``, plus a private symbol map and a manifest.

Neither step overwrites existing output unless ``force`` is true.

Source: the legacy CRSP daily (SIZ) tables on WRDS, ``crsp.dsf``, ``crsp.dsenames``,
``crsp.dsedelist``, and ``crsp.dsi``. CRSP has moved to the CIZ format, so the
fetch checks the available coverage and rejects windows outside it.
"""

import csv
import hashlib
import json
import math
import platform
import random
import re
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from forecast_agents.tasks.panel import Bar, Panel

SOURCE = "WRDS CRSP legacy daily stock files (SIZ): crsp.dsf, dsenames, dsedelist, dsi"
DAILY_COLUMNS = ("permno", "permco", "date", "ret", "retx", "prc", "vol", "shrout",
                 "cfacpr", "cfacshr", "openprc", "bidlo", "askhi", "bid", "ask")
NAME_COLUMNS = ("permno", "permco", "namedt", "nameendt", "shrcd", "exchcd", "siccd",
                "ticker", "comnam")
DELIST_COLUMNS = ("permno", "dlstdt", "dlstcd", "dlret", "dlretx")
MARKET_COLUMNS = ("date", "vwretd", "vwretx", "ewretd", "ewretx", "sprtrn", "spindx")
TABLES = {"crsp.dsf": DAILY_COLUMNS, "crsp.dsenames": NAME_COLUMNS,
          "crsp.dsedelist": DELIST_COLUMNS, "crsp.dsi": MARKET_COLUMNS}
UNIVERSE_COLUMNS = ("formation_rank", "permno", "permco", "company_mktcap", "mktcap", "prc",
                    "shrout", "shrcd", "exchcd", "siccd", "ticker", "comnam")
SYMBOL_COLUMNS = ("symbol", "permno", "permco", "formation_rank", "company_mktcap",
                  "ticker", "comnam", "shrcd", "exchcd", "siccd")

# Each query is sent with a "/* crsp:<name> */" tag, which identifies it in tests.
QUERIES = {
    "coverage": "select min(date) as first_date, max(date) as last_date from crsp.dsi",
    "formation": "select max(date) as formation_date from crsp.dsi where date < %(start)s",
    "session": "select count(*) as sessions from crsp.dsi where date = %(day)s",
    "universe": (
        "select d.permno, d.permco, d.prc, d.shrout, n.shrcd, n.exchcd, n.siccd, n.ticker, n.comnam "
        "from crsp.dsf as d join crsp.dsenames as n "
        "on n.permno = d.permno and d.date between n.namedt and n.nameendt "
        "where d.date = %(formation_date)s "
        "and n.shrcd in %(share_codes)s and n.exchcd in %(exchange_codes)s"),
    "daily": (f"select {', '.join(DAILY_COLUMNS)} from crsp.dsf "
              "where permno in %(permnos)s and date between %(start)s and %(end)s "
              "order by permno, date"),
    "names": (f"select {', '.join(NAME_COLUMNS)} from crsp.dsenames "
              "where permno in %(permnos)s and nameendt >= %(start)s and namedt <= %(end)s "
              "order by permno, namedt"),
    "delist": f"select {', '.join(DELIST_COLUMNS)} from crsp.dsedelist where permno in %(permnos)s order by permno",
    "market": (f"select {', '.join(MARKET_COLUMNS)} from crsp.dsi "
               "where date between %(start)s and %(end)s order by date"),
}
SPEC_LAYOUT = {
    "window": ("start", "end", "formation_date"),
    "universe": ("candidates", "stocks", "share_codes", "exchange_codes", "one_security_per_company"),
    "panel": ("mode", "anonymize", "seed"),
    "output": ("raw_dir", "panel"),
}


def _is_iso_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


@dataclass(frozen=True)
class DatasetSpec:
    """One experiment window: what to fetch from WRDS and how to build the panel."""

    name: str
    start: str
    end: str
    raw_dir: str
    panel: str
    formation_date: str = ""
    candidates: int = 100
    stocks: int = 30
    share_codes: tuple[int, ...] = (10, 11)
    exchange_codes: tuple[int, ...] = (1, 2, 3)
    one_security_per_company: bool = True
    mode: str = "complete"
    anonymize: bool = True
    seed: int = 7652

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name may contain only letters, digits, '.', '_', and '-'")
        if not _is_iso_date(self.start) or not _is_iso_date(self.end) or self.start >= self.end:
            raise ValueError("window.start and window.end must be YYYY-MM-DD dates with start < end")
        if self.formation_date and (not _is_iso_date(self.formation_date) or self.formation_date >= self.start):
            raise ValueError("window.formation_date must be a YYYY-MM-DD date before window.start")
        if any(type(getattr(self, key)) is not int for key in ("candidates", "stocks", "seed")):
            raise ValueError("universe.candidates, universe.stocks, and panel.seed must be integers")
        if not 1 <= self.stocks <= self.candidates:
            raise ValueError("Need 1 <= universe.stocks <= universe.candidates")
        for key in ("share_codes", "exchange_codes"):
            codes = getattr(self, key)
            if not isinstance(codes, tuple) or not codes or any(type(code) is not int for code in codes):
                raise ValueError(f"universe.{key} must be a nonempty list of integers")
        if type(self.one_security_per_company) is not bool or type(self.anonymize) is not bool:
            raise ValueError("universe.one_security_per_company and panel.anonymize must be booleans")
        if self.mode != "complete":
            raise ValueError('panel.mode must be "complete": the current loader needs a balanced panel')
        if not isinstance(self.raw_dir, str) or not self.raw_dir:
            raise ValueError("output.raw_dir is required")
        if not isinstance(self.panel, str) or not self.panel.endswith(".json"):
            raise ValueError("output.panel must be a .json path")


def load_spec(path):
    """Read a dataset spec TOML (see configs/datasets/)."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - {"name", *SPEC_LAYOUT}
    if unknown:
        raise ValueError(f"Unknown dataset spec keys: {sorted(unknown)}")
    values = {"name": raw.get("name")}
    for section, keys in SPEC_LAYOUT.items():
        table = raw.get(section, {})
        if not isinstance(table, dict) or set(table) - set(keys):
            raise ValueError(f"Unknown keys in [{section}]; allowed: {', '.join(keys)}")
        values.update(table)
    for key in ("start", "end", "formation_date"):
        if type(values.get(key)) is date:  # unquoted TOML date
            values[key] = values[key].isoformat()
    for key in ("share_codes", "exchange_codes"):
        if isinstance(values.get(key), list):
            values[key] = tuple(values[key])
    try:
        return DatasetSpec(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid dataset spec {path}: {exc}") from None


def _clean(value):
    # SQL NULL arrives as None, NaN, NaT, or pandas.NA depending on the column dtype.
    try:
        missing = bool(value is None or value != value)
    except TypeError:  # pandas.NA refuses boolean conversion
        return None
    return None if missing else value


def _records(frame):
    return [{key: _clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _query(db, name, params=None, sql=None):
    return _records(db.raw_sql(f"/* crsp:{name} */ {sql or QUERIES[name]}", params=params))


def _number(value):
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _iso(value):
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value)[:10]
    date.fromisoformat(text)
    return text


def _cell(value):
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        # WRDS returns some identifiers (PERMNO, codes) as floats.
        return str(int(value)) if value.is_integer() else repr(value)
    return str(value)


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([_cell(row.get(column)) for column in columns] for row in rows)


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_text(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path, value):
    _write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _versions():
    from importlib import metadata
    versions = {"python": platform.python_version()}
    for package in ("wrds", "pandas"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def rank_universe(rows, candidates, one_security_per_company=True):
    """Rank formation-date securities by market cap, using formation-date data only.

    No survival or future-data filter is applied here. Market cap is |PRC| x SHROUT
    in thousands of dollars; a negative PRC is CRSP's bid/ask midpoint. With one
    security per company, a company's total cap ranks it and its largest share
    class represents it.
    """
    securities = {}
    for row in rows:
        prc, shrout = _number(row.get("prc")), _number(row.get("shrout"))
        if prc is None or shrout is None or prc == 0 or shrout <= 0:
            continue
        permno = int(row["permno"])
        if permno not in securities:  # overlapping name records: keep the first
            securities[permno] = {**row, "permno": permno, "permco": int(row["permco"]),
                                  "prc": abs(prc), "shrout": shrout, "mktcap": abs(prc) * shrout}
    if one_security_per_company:
        companies = defaultdict(list)
        for security in securities.values():
            companies[security["permco"]].append(security)
        ranked = [{**max(group, key=lambda s: (s["mktcap"], -s["permno"])),
                   "company_mktcap": sum(s["mktcap"] for s in group)}
                  for group in companies.values()]
    else:
        ranked = [{**security, "company_mktcap": security["mktcap"]} for security in securities.values()]
    ranked.sort(key=lambda s: (-s["company_mktcap"], s["permno"]))
    return [{**security, "formation_rank": rank} for rank, security in enumerate(ranked[:candidates], 1)]


def fetch_raw(spec, db, force=False):
    """Download one window's raw CRSP extract through a ``wrds.Connection``-like ``db``."""
    raw_dir = Path(spec.raw_dir)
    if raw_dir.exists() and not force:
        raise FileExistsError(f"{raw_dir} exists; pass --force to replace it")
    for table, columns in TABLES.items():
        try:
            _query(db, "preflight", sql=f"select {', '.join(columns)} from {table} limit 0")
        except Exception as exc:  # driver errors vary; report them with guidance
            raise RuntimeError(f"Cannot read {table} ({', '.join(columns)}). Check your WRDS CRSP "
                               f"subscription and the table schema. Original error: {exc}") from exc
    coverage = _query(db, "coverage")[0]
    first, last = _iso(coverage["first_date"]), _iso(coverage["last_date"])
    if spec.start < first or spec.end > last:
        raise ValueError(f"CRSP daily data on WRDS cover {first} to {last}; "
                         f"the window {spec.start} to {spec.end} is outside that range")
    start, end = date.fromisoformat(spec.start), date.fromisoformat(spec.end)
    if spec.formation_date:
        formation = spec.formation_date
        if not _query(db, "session", {"day": date.fromisoformat(formation)})[0]["sessions"]:
            raise ValueError(f"formation_date {formation} is not a CRSP trading session")
    else:
        formation = _iso(_query(db, "formation", {"start": start})[0]["formation_date"])
        if not formation:
            raise ValueError("No CRSP trading session precedes window.start")
    cross_section = _query(db, "universe", {"formation_date": date.fromisoformat(formation),
                                            "share_codes": spec.share_codes,
                                            "exchange_codes": spec.exchange_codes})
    universe = rank_universe(cross_section, spec.candidates, spec.one_security_per_company)
    if len(universe) < spec.stocks:
        raise ValueError(f"Only {len(universe)} eligible companies on {formation}; lower universe.stocks")
    permnos = tuple(sorted(row["permno"] for row in universe))
    window = {"permnos": permnos, "start": start, "end": end}
    tables = {"universe": (UNIVERSE_COLUMNS, universe),
              "daily": (DAILY_COLUMNS, _query(db, "daily", window)),
              "names": (NAME_COLUMNS, _query(db, "names", window)),
              "delist": (DELIST_COLUMNS, _query(db, "delist", {"permnos": permnos})),
              "market": (MARKET_COLUMNS, _query(db, "market", {"start": start, "end": end}))}
    # Write only after every query succeeded, so a failure leaves no partial extract.
    raw_dir.mkdir(parents=True, exist_ok=force)
    for name, (columns, rows) in tables.items():
        _write_csv(raw_dir / f"{name}.csv", columns, rows)
    manifest = {
        "dataset": spec.name, "source": SOURCE, "spec": asdict(spec), "fetched_at": _now(),
        "crsp_coverage": {"first": first, "last": last}, "formation_date": formation,
        "universe_rule": (f"Top {spec.candidates} companies by market cap on {formation} among "
                          f"SHRCD {list(spec.share_codes)} and EXCHCD {list(spec.exchange_codes)}; "
                          "no survival filter. universe.csv prc is |PRC|; mktcap is in $1000s."),
        "formation_cross_section": len(cross_section),
        "rows": {name: len(rows) for name, (_, rows) in tables.items()},
        "tables": {table: list(columns) for table, columns in TABLES.items()},
        "software": _versions(),
    }
    _write_json(raw_dir / "manifest.json", manifest)
    return {"raw_dir": str(raw_dir), "formation_date": formation, "crsp_coverage": manifest["crsp_coverage"],
            "rows": manifest["rows"]}


def _window_returns(rows_by_date, calendar):
    """Daily total returns for sessions 2..n, or the reason a stock is incomplete."""
    missing = [day for day in calendar if day not in rows_by_date]
    if missing:
        return None, f"no CRSP row on {len(missing)} of {len(calendar)} sessions (first {missing[0]})"
    returns, invalid = [], []
    for day in calendar[1:]:
        value = _number(rows_by_date[day]["ret"])
        # Missing returns, and legacy negative codes (-66 to -99), are not returns.
        if value is None or value <= -1:
            invalid.append(day)
        returns.append(value)
    if invalid:
        return None, f"missing or invalid RET on {len(invalid)} sessions (first {invalid[0]})"
    return returns, ""


def total_return_index(returns):
    """Total-return index compounded forward from 1.0, so each level uses past data only.

    Back-adjusted prices would leak future splits and dividends into earlier levels.
    """
    levels = [1.0]
    for value in returns:
        levels.append(levels[-1] * (1.0 + value))
    return levels


def _panel_outputs(panel_path):
    stem = panel_path.name[:-len(".json")]
    return {"panel": panel_path, "symbols": panel_path.with_name(stem + ".symbols.csv"),
            "manifest": panel_path.with_name(stem + ".manifest.json")}


def _check_extract(spec, raw_manifest):
    fetched = raw_manifest["spec"]
    for key in ("start", "end", "share_codes", "exchange_codes", "one_security_per_company"):
        wanted = getattr(spec, key)
        if (list(wanted) if isinstance(wanted, tuple) else wanted) != fetched[key]:
            raise ValueError(f"{spec.raw_dir} was fetched with {key}={fetched[key]!r}, but the spec "
                             f"now says {wanted!r}; fetch again with --force")
    if spec.candidates > fetched["candidates"]:
        raise ValueError(f"{spec.raw_dir} holds {fetched['candidates']} candidates; fetch again with --force")
    if spec.formation_date and spec.formation_date != raw_manifest["formation_date"]:
        raise ValueError(f"{spec.raw_dir} was formed on {raw_manifest['formation_date']}; fetch again with --force")


def build_panel(spec, force=False):
    """Build the panel JSON, a private symbol map, and a manifest from a raw extract."""
    raw_dir = Path(spec.raw_dir)
    outputs = _panel_outputs(Path(spec.panel))
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{existing[0]} exists; pass --force to replace it")
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No raw extract at {raw_dir}; run the fetch step first")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_extract(spec, raw_manifest)
    calendar = sorted({row["date"] for row in _read_csv(raw_dir / "market.csv")
                       if spec.start <= row["date"] <= spec.end})
    if len(calendar) < 2:
        raise ValueError("The raw extract has fewer than two trading sessions in the window")
    daily = defaultdict(dict)
    for row in _read_csv(raw_dir / "daily.csv"):
        permno, day = int(row["permno"]), row["date"]
        if day in daily[permno]:
            raise ValueError(f"Duplicate CRSP row for PERMNO {permno} on {day}")
        daily[permno][day] = row
    candidates = sorted(_read_csv(raw_dir / "universe.csv"),
                        key=lambda row: int(row["formation_rank"]))[:spec.candidates]
    selected, dropped = [], []
    for candidate in candidates:
        if len(selected) == spec.stocks:
            break
        returns, reason = _window_returns(daily.get(int(candidate["permno"]), {}), calendar)
        if reason:
            dropped.append({"formation_rank": int(candidate["formation_rank"]),
                            "permno": int(candidate["permno"]), "ticker": candidate["ticker"],
                            "reason": reason})
        else:
            selected.append((candidate, returns))
    if len(selected) < spec.stocks:
        raise ValueError(f"Only {len(selected)} of {len(candidates)} candidates have complete data from "
                         f"{calendar[0]} to {calendar[-1]}; raise universe.candidates and fetch again, "
                         "or lower universe.stocks")
    order = list(selected)
    if spec.anonymize:
        random.Random(spec.seed).shuffle(order)
        symbols = [f"S{index:03}" for index in range(len(order))]
    else:
        symbols = [str(int(candidate["permno"])) for candidate, _ in order]
    levels = {symbol: total_return_index(returns) for symbol, (_, returns) in zip(symbols, order)}
    rows = [{"symbol": symbol, "date": day, "close": levels[symbol][index]}
            for index, day in enumerate(calendar) for symbol in sorted(levels)]
    text = "[\n" + ",\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n]\n"
    panel = Panel([Bar(**row) for row in json.loads(text)])  # exactly what the run loader does
    symbol_rows = [{**candidate, "symbol": symbol} for symbol, (candidate, _) in zip(symbols, order)]
    manifest = {
        "dataset": spec.name, "built_at": _now(), "spec": asdict(spec),
        "raw_extract": {key: raw_manifest[key] for key in ("source", "fetched_at", "formation_date", "crsp_coverage")},
        "close_definition": ("Total-return index from CRSP daily RET (dividends and splits included), "
                             "compounded forward from 1.0 on the first session."),
        "universe_rule": (f"The {spec.stocks} highest-ranked companies by market cap on "
                          f"{raw_manifest['formation_date']} with complete data in the window."),
        "survivorship": ("Candidates lacking a CRSP row or a valid RET on any session were skipped "
                         "(see dropped). This conditions on survival through the window end and "
                         "biases the panel toward survivors."),
        "sessions": {"count": len(panel.dates), "first": panel.dates[0], "last": panel.dates[-1]},
        "stocks": len(panel.symbols), "dropped": dropped,
        "files": {"panel": str(outputs["panel"]),
                  "panel_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "symbol_map": str(outputs["symbols"])},
    }
    outputs["panel"].parent.mkdir(parents=True, exist_ok=True)
    _write_text(outputs["panel"], text)
    _write_csv(outputs["symbols"], SYMBOL_COLUMNS, symbol_rows)
    _write_json(outputs["manifest"], manifest)
    return {"panel": str(outputs["panel"]), "stocks": len(panel.symbols),
            "sessions": manifest["sessions"], "dropped": len(dropped),
            "symbol_map": str(outputs["symbols"]), "manifest": str(outputs["manifest"])}
