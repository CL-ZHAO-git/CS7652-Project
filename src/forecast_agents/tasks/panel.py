import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Bar:
    symbol: str
    date: str
    close: float


@dataclass(frozen=True)
class TaskSpec:
    train_end: str
    validation_end: str
    horizons: tuple[int, ...] = (1, 3, 5)

    def __post_init__(self):
        if any(date.fromisoformat(day).isoformat() != day for day in (self.train_end, self.validation_end)):
            raise ValueError("Use canonical YYYY-MM-DD dates")
        if self.train_end >= self.validation_end:
            raise ValueError("Training must precede validation")
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Horizons must be nonempty and unique")
        if any(type(h) is not int or h <= 0 for h in self.horizons):
            raise ValueError("Horizons must be positive integers")


class Panel:
    """Balanced observed trading sessions; no calendar-day or missing-row shifting."""

    def __init__(self, bars: list[Bar]):
        if not bars:
            raise ValueError("Empty panel")
        groups = defaultdict(dict)
        for bar in bars:
            if date.fromisoformat(bar.date).isoformat() != bar.date:
                raise ValueError("Use canonical YYYY-MM-DD dates")
            if (not isinstance(bar.symbol, str) or not bar.symbol
                    or type(bar.close) not in (int, float) or not math.isfinite(bar.close) or bar.close <= 0):
                raise ValueError("Symbols and positive finite closes are required")
            if bar.date in groups[bar.symbol]:
                raise ValueError("Duplicate stock-date observation")
            groups[bar.symbol][bar.date] = bar.close
        self.dates = tuple(sorted({bar.date for bar in bars}))
        self.symbols = tuple(sorted(groups))
        for values in groups.values():
            if set(values) != set(self.dates):
                raise ValueError("Initial task loader requires a balanced panel; do not skip missing sessions")
        self.prices = {symbol: tuple(groups[symbol][day] for day in self.dates)
                       for symbol in self.symbols}
        self._index = {day: i for i, day in enumerate(self.dates)}

    def history(self, cutoff: str):
        if cutoff not in self._index:
            raise ValueError("Cutoff must be an observed trading session")
        end = self._index[cutoff]
        return [{"symbol": symbol, "date": day, "close": self.prices[symbol][i]}
                for i, day in enumerate(self.dates[:end + 1]) for symbol in self.symbols]

    def label(self, symbol: str, origin: str, horizon: int, available_by: str):
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        if available_by not in self._index:
            raise ValueError("Information cutoff must be an observed trading session")
        i = self._index[origin]
        end = i + horizon
        if end >= len(self.dates) or self.dates[end] > available_by:
            raise ValueError("Label has not matured by the information cutoff")
        return self.prices[symbol][end] / self.prices[symbol][i] - 1

    def origins(self, task: TaskSpec, split: str):
        if task.train_end not in self._index or task.validation_end not in self._index:
            raise ValueError("Task boundaries must be observed trading sessions")
        if split == "validation":
            start, end = task.train_end, task.validation_end
        elif split == "test":
            start, end = task.validation_end, self.dates[-1]
        else:
            raise ValueError("Split must be validation or test")
        origins = tuple(day for i, day in enumerate(self.dates)
                        if start < day <= end and i + max(task.horizons) < len(self.dates)
                        and self.dates[i + max(task.horizons)] <= end)
        if not origins:
            raise ValueError(f"No fully matured {split} origins")
        return origins


def synthetic_panel(stocks=4, sessions=120, seed=7):
    if stocks < 1 or sessions < 20:
        raise ValueError("Need at least one stock and twenty sessions")
    rng = random.Random(seed)
    dates = []
    day = date(2020, 1, 1)
    while len(dates) < sessions:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)
    prices = [100.0 + i * 10 for i in range(stocks)]
    previous = [0.0] * stocks
    rows = []
    for day in dates:
        market = rng.gauss(0, 0.006)
        for i in range(stocks):
            ret = 0.0002 + 0.15 * previous[i] + market + rng.gauss(0, 0.009)
            prices[i] *= 1 + ret
            previous[i] = ret
            rows.append(Bar(f"S{i:03}", day, prices[i]))
    return Panel(rows)
