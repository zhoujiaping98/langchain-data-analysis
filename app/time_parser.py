from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass


@dataclass
class TimeRange:
    start: str  # YYYY-MM-DD
    end_exclusive: str  # YYYY-MM-DD
    label: str


def infer_time_range(query: str) -> TimeRange | None:
    q = query
    today = dt.date.today()

    if re.search(r"上周", q):
        start = today - dt.timedelta(days=today.weekday() + 7)
        end = start + dt.timedelta(days=7)
        return TimeRange(start.isoformat(), end.isoformat(), "last_week")

    if re.search(r"本周", q):
        start = today - dt.timedelta(days=today.weekday())
        end = start + dt.timedelta(days=7)
        return TimeRange(start.isoformat(), end.isoformat(), "this_week")

    if re.search(r"上个月", q):
        first_this = today.replace(day=1)
        last_prev = first_this - dt.timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return TimeRange(first_prev.isoformat(), first_this.isoformat(), "last_month")

    if re.search(r"本月", q):
        first_this = today.replace(day=1)
        next_month = (first_this.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        return TimeRange(first_this.isoformat(), next_month.isoformat(), "this_month")

    m = re.search(r"最近\s*(\d+)\s*天|近\s*(\d+)\s*天", q)
    if m:
        n = int(m.group(1) or m.group(2))
        start = today - dt.timedelta(days=n)
        return TimeRange(start.isoformat(), today.isoformat(), f"last_{n}_days")

    dates = re.findall(r"(\d{4}-\d{2}-\d{2})", q)
    if len(dates) >= 2:
        return TimeRange(dates[0], dates[1], "custom")

    return None