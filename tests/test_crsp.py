import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from forecast_agents.cli import run_experiment
from forecast_agents.config import DataConfig, ExperimentConfig, load_config
from forecast_agents.datasets.crsp import DatasetSpec, build_panel, fetch_raw, load_spec, rank_universe
from forecast_agents.tasks.panel import Bar, Panel

ROOT = Path(__file__).resolve().parents[1]


def weekdays(start, count):
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


CALENDAR = weekdays(date(2020, 12, 28), 42)  # CALENDAR[0] becomes the formation date


class FakeNA:
    """Behaves like pandas.NA: comparisons return NA and bool() raises."""

    def __ne__(self, other):
        return self

    __eq__ = __ne__

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return [dict(row) for row in self.rows]


def company(permno, permco, prc, shrout, ticker):
    # WRDS returns identifiers and codes as floats.
    return {"permno": float(permno), "permco": float(permco), "prc": prc, "shrout": shrout,
            "shrcd": 11.0, "exchcd": 1.0, "siccd": 3571.0, "ticker": ticker, "comnam": f"{ticker} CORP, INC"}


COMPANIES = [
    company(101, 1, 100.0, 5000.0, "TICKA"),  # rank 1, together with its second share class
    company(102, 2, -80.0, 5000.0, "TICKB"),  # rank 2; a negative PRC is a bid/ask midpoint
    company(103, 3, 60.0, 5000.0, "TICKC"),
    company(104, 4, 40.0, 5000.0, "TICKD"),
    company(105, 5, 20.0, 5000.0, "TICKE"),   # rank 5, beyond four candidates
    company(106, 1, 10.0, 1000.0, "TICKF"),   # second share class of company 1
    company(107, 7, None, 5000.0, "TICKG"),   # no price on the formation date
]
GAPS = {102: {CALENDAR[20]}}             # no CRSP row that day (halt, delisting, ...)
MISSING_RETURNS = {103: {CALENDAR[30]}}  # row present, RET missing


def daily_return(permno, index):
    return round(0.002 * ((permno + 3 * index) % 7 - 3), 6)


class FakeWRDS:
    """Answers the fetch queries the way wrds.Connection.raw_sql(sql, params=...) does."""

    def __init__(self, companies=COMPANIES, gaps=GAPS, missing_returns=MISSING_RETURNS,
                 last=date(2024, 12, 31)):
        self.companies, self.gaps, self.missing_returns, self.last = companies, gaps, missing_returns, last
        self.calls = []

    def raw_sql(self, sql, params=None, **options):
        name = re.match(r"/\* crsp:(\w+) \*/", sql).group(1)
        self.calls.append((name, params))
        params = params or {}
        wanted = [c for c in self.companies if int(c["permno"]) in params.get("permnos", ())]
        in_window = [(i, day) for i, day in enumerate(CALENDAR)
                     if params.get("start", CALENDAR[0]) <= day <= params.get("end", CALENDAR[-1])]
        if name == "preflight":
            rows = []
        elif name == "coverage":
            rows = [{"first_date": CALENDAR[0], "last_date": self.last}]
        elif name == "formation":
            rows = [{"formation_date": max(day for day in CALENDAR if day < params["start"])}]
        elif name == "session":
            rows = [{"sessions": int(params["day"] in CALENDAR)}]
        elif name == "universe":
            rows = self.companies
        elif name == "daily":
            rows = [self.daily_row(c, i, day) for c in wanted for i, day in in_window
                    if day not in self.gaps.get(int(c["permno"]), ())]
        elif name == "names":
            rows = [{**c, "namedt": date(2000, 1, 3), "nameendt": self.last} for c in wanted]
        elif name == "delist":
            rows = [{"permno": c["permno"], "dlstdt": self.last, "dlstcd": 100.0,
                     "dlret": FakeNA(), "dlretx": None} for c in wanted]
        elif name == "market":
            rows = [{"date": day, "vwretd": 0.001, "vwretx": 0.0009, "ewretd": 0.0012,
                     "ewretx": 0.001, "sprtrn": 0.0008, "spindx": 3700.0 + i} for i, day in in_window]
        else:
            raise AssertionError(f"Unexpected query {name}")
        return FakeFrame(rows)

    def daily_row(self, company, index, day):
        permno, price = int(company["permno"]), abs(company["prc"])
        ret = FakeNA() if day in self.missing_returns.get(permno, ()) else daily_return(permno, index)
        return {"permno": company["permno"], "permco": company["permco"], "date": day, "ret": ret,
                "retx": ret, "prc": price, "vol": 1000.0, "shrout": company["shrout"], "cfacpr": 1.0,
                "cfacshr": 1.0, "openprc": price, "bidlo": price - 1, "askhi": price + 1,
                "bid": FakeNA(), "ask": None}


def make_spec(directory, **changes):
    values = {"name": "test_panel", "start": CALENDAR[1].isoformat(), "end": CALENDAR[-1].isoformat(),
              "raw_dir": str(Path(directory) / "raw"), "panel": str(Path(directory) / "panels/test.json"),
              "candidates": 4, "stocks": 2, **changes}
    return DatasetSpec(**values)


class CrspTests(unittest.TestCase):
    def test_universe_is_point_in_time_with_one_security_per_company(self):
        ranked = rank_universe(COMPANIES, candidates=10)
        self.assertEqual([row["permno"] for row in ranked], [101, 102, 103, 104, 105])
        self.assertEqual([row["formation_rank"] for row in ranked], [1, 2, 3, 4, 5])
        self.assertEqual(ranked[0]["company_mktcap"], 100.0 * 5000 + 10.0 * 1000)
        self.assertEqual(ranked[1]["prc"], 80.0)
        securities = [row["permno"] for row in rank_universe(COMPANIES, 10, one_security_per_company=False)]
        self.assertEqual(securities, [101, 102, 103, 104, 105, 106])

    def test_fetch_and_build_make_a_balanced_total_return_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = make_spec(directory)
            db = FakeWRDS()
            fetch_raw(spec, db)
            daily = next(params for name, params in db.calls if name == "daily")
            self.assertEqual(daily["permnos"], (101, 102, 103, 104))  # top four companies only
            self.assertEqual((daily["start"], daily["end"]), (CALENDAR[1], CALENDAR[-1]))
            with (Path(spec.raw_dir) / "daily.csv").open(encoding="utf-8", newline="") as stream:
                first = next(csv.DictReader(stream))
            self.assertEqual((first["permno"], first["date"], first["bid"], first["ask"]),
                             ("101", CALENDAR[1].isoformat(), "", ""))

            summary = build_panel(spec)
            self.assertEqual((summary["stocks"], summary["dropped"]), (2, 2))
            manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
            reasons = {row["permno"]: row["reason"] for row in manifest["dropped"]}
            self.assertIn("no CRSP row", reasons[102])
            self.assertIn("RET", reasons[103])
            text = Path(spec.panel).read_text(encoding="utf-8")
            self.assertNotIn("TICK", text)  # anonymized
            rows = json.loads(text)
            self.assertTrue(all(set(row) == {"symbol", "date", "close"} for row in rows))
            panel = Panel([Bar(**row) for row in rows])
            self.assertEqual(panel.symbols, ("S000", "S001"))
            self.assertEqual(panel.dates[0], CALENDAR[1].isoformat())
            with Path(summary["symbol_map"]).open(encoding="utf-8", newline="") as stream:
                permno_of = {row["symbol"]: int(row["permno"]) for row in csv.DictReader(stream)}
            self.assertEqual(set(permno_of.values()), {101, 104})
            for symbol, permno in permno_of.items():
                # Panel labels are compounded CRSP returns: sessions 2..6 of CALENDAR.
                expected = math.prod(1 + daily_return(permno, i) for i in range(2, 7)) - 1
                self.assertAlmostEqual(panel.label(symbol, panel.dates[0], 5, panel.dates[-1]), expected)

    def test_build_is_deterministic_and_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = make_spec(directory)
            fetch_raw(spec, FakeWRDS())
            with self.assertRaises(FileExistsError):
                fetch_raw(spec, FakeWRDS())
            build_panel(spec)
            first = Path(spec.panel).read_bytes()
            with self.assertRaises(FileExistsError):
                build_panel(spec)
            build_panel(spec, force=True)
            self.assertEqual(Path(spec.panel).read_bytes(), first)
            self.assertNotIn(b"\r\n", first)

    def test_window_outside_crsp_coverage_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = make_spec(directory)
            with self.assertRaisesRegex(ValueError, "cover"):
                fetch_raw(spec, FakeWRDS(last=CALENDAR[20]))
            self.assertFalse(Path(spec.raw_dir).exists())

    def test_build_checks_the_extract_and_enough_complete_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = make_spec(directory)
            fetch_raw(spec, FakeWRDS())
            with self.assertRaisesRegex(ValueError, "fetched"):
                build_panel(replace(spec, start=CALENDAR[2].isoformat()))
            with self.assertRaisesRegex(ValueError, "complete data"):
                build_panel(replace(spec, stocks=3))

    def test_specs_reject_unknown_keys_and_accept_toml_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.toml"
            body = ('name = "x"\n[window]\nstart = 2017-01-03\nend = 2019-06-28\n'
                    '[output]\nraw_dir = "data/raw/x"\npanel = "data/panels/x.json"\n')
            path.write_text(body, encoding="utf-8")
            self.assertEqual(load_spec(path).start, "2017-01-03")
            path.write_text(body + "[universe]\nstock = 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_spec(path)
        shipped = load_spec(ROOT / "configs/datasets/crsp_lc30_2017_2019.toml")
        self.assertEqual((shipped.candidates, shipped.stocks), (100, 30))

    def test_build_script_runs_from_the_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = make_spec(directory)
            fetch_raw(spec, FakeWRDS())
            path = Path(directory) / "spec.toml"
            path.write_text(f'name = "{spec.name}"\n[window]\nstart = "{spec.start}"\nend = "{spec.end}"\n'
                            '[universe]\ncandidates = 4\nstocks = 2\n'
                            f'[output]\nraw_dir = "{Path(spec.raw_dir).as_posix()}"\n'
                            f'panel = "{Path(spec.panel).as_posix()}"\n', encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/crsp_data.py"), "build", str(path)],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["stocks"], 2)

    def test_runs_use_calendar_boundaries_on_a_built_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = make_spec(directory)
            fetch_raw(spec, FakeWRDS())
            build_panel(spec)
            saturday = CALENDAR[16] + timedelta(days=(5 - CALENDAR[16].weekday()) % 7)
            data = DataConfig(source="json", path=spec.panel, train_end=saturday.isoformat(),
                              validation_end=CALENDAR[30].isoformat())
            output = Path(directory) / "run"
            summary = run_experiment(ExperimentConfig(mode="single", data=data), output)
            self.assertEqual(summary["status"], "completed", summary.get("error"))
            task = json.loads((output / "workspace/task.json").read_text(encoding="utf-8"))
            self.assertEqual(task["train_end"], max(day for day in CALENDAR if day <= saturday).isoformat())
            report = json.loads((output / "test_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["rows"])
            self.assertTrue(all(row["origin"] > task["validation_end"] for row in report["rows"]))

    def test_configs_take_dates_or_session_counts_not_both(self):
        config = load_config(ROOT / "configs/crsp_demo.toml")
        self.assertEqual((config.data.train_end, config.data.validation_end), ("2018-12-31", "2019-03-29"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for body in ('[data]\ntrain_end = "2020-01-31"\nvalidation_end = "2020-03-31"\ntrain_sessions = 5\n',
                         '[data]\ntrain_end = "2020-01-31"\n',
                         '[data]\ntrain_end = "2020-03-31"\nvalidation_end = "2020-01-31"\n'):
                path.write_text(body, encoding="utf-8")
                with self.subTest(body=body), self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
