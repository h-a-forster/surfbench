"""Offline tests of the Tardis extractor and the Deribit live adapter (DESIGN 7.4).

No test touches the network: a fake opener serves synthetic gzip CSVs in the Tardis layout.
"""

from __future__ import annotations

import email.message
import gzip
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from surfbench.data import deribit, tardis
from surfbench.quotes import file_sha256, read_snapshot

MONTH = "2026-10"


class FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self._buf = io.BytesIO(data)
        self.status = 200
        self.headers = headers or {"x-dataset-size": str(len(data)), "x-md5": "abc"}
        self.bytes_read = 0

    def read(self, n: int = -1) -> bytes:
        out = self._buf.read(n)
        self.bytes_read += len(out)
        return out

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    def __init__(self, plan: list[Any]) -> None:
        self.plan = list(plan)
        self.calls: list[str] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, req: Any, timeout: float = 0) -> FakeResponse:
        self.calls.append(req.full_url)
        assert req.get_method() == "GET" and not req.has_header("Range")
        item = self.plan.pop(0)
        if isinstance(item, BaseException):
            raise item
        resp = FakeResponse(item)
        self.responses.append(resp)
        return resp


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.invalid", code, "err", hdrs, None)


def _row(symbol: str, ts: int, bid: str = "0.01", ask: str = "0.012", und: str = "60000") -> str:
    typ = "call" if symbol.endswith("C") else "put"
    strike = symbol.split("-")[2] if symbol.count("-") >= 3 else "0"
    vals = [
        "deribit",
        symbol,
        str(ts - 5),
        str(ts),
        typ,
        strike,
        "1793001600000000",
        "10",
        "",
        bid,
        "1",
        "50.1",
        ask,
        "2",
        "52.3",
        "0.011",
        "51.0",
        "SYN.BTC-30OCT26",
        und,
        "0.5",
        "0.0",
        "1.0",
        "-1.0",
        "0.0",
    ]
    return ",".join(vals)


def _file(t0_us: int, t1_us: int, *, extra_minutes: int = 200, bad: str | None = None) -> bytes:
    lines = [",".join(tardis.TARDIS_COLUMNS)]
    base = t0_us - 60_000_000  # midnight
    # State dump at midnight: two BTC inverse options, one linear option (dropped), one ETH.
    lines.append(_row("BTC-30OCT26-60000-C", base + 100))
    lines.append(_row("BTC-30OCT26-55000-P", base + 200))
    lines.append(_row("BTC_USDC-30OCT26-60000-C", base + 300))
    lines.append(_row("ETH-30OCT26-3000-C", base + 400, und="3000"))
    # Update before t0 (bid moves), then after t0 and before t1, then much later rows.
    lines.append(_row("BTC-30OCT26-60000-C", t0_us - 10, bid="0.0105"))
    lines.append(_row("BTC-30OCT26-60000-C", t0_us + 10, bid="0.0110"))
    lines.append(_row("BTC-30OCT26-65000-C", t0_us + 20))  # listed between t0 and t1
    if bad == "columns":
        lines.append("deribit,BTC-30OCT26-60000-C,1,2")
    if bad == "order":
        lines.append(_row("BTC-30OCT26-60000-C", t0_us - 1000))
    if bad == "number":
        lines.append(_row("BTC-30OCT26-60000-C", t0_us + 30, bid="abc"))
    for i in range(extra_minutes * 60):
        lines.append(_row("BTC-30OCT26-60000-C", t1_us + 1000 + i * 1_000_000))
    return gzip.compress(("\n".join(lines) + "\n").encode("ascii"))


def _cfg(tmp_path: Path, **kw: Any) -> tardis.TardisConfig:
    base = {
        "cache_dir": tmp_path / "cache",
        "months": (MONTH,),
        "horizon_min": 15,
        "request_gap_s": 0,
    }
    base.update(kw)
    return tardis.TardisConfig(**base)


def test_extract_month_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tardis, "CHUNK", 1024)
    cfg = _cfg(tmp_path)
    t0, t1 = cfg.times_us(MONTH)
    data = _file(t0, t1)
    opener = FakeOpener([data])
    meta = tardis.extract_month(cfg, MONTH, opener=opener)
    assert meta["status"] == "ok", meta
    d = cfg.month_dir(MONTH)
    s0 = read_snapshot(d / "snap_t0.csv.gz")
    s1 = read_snapshot(d / "snap_t1.csv.gz")
    assert list(s0.symbol) == ["BTC-30OCT26-55000-P", "BTC-30OCT26-60000-C", "ETH-30OCT26-3000-C"]
    assert s0.bid_price[1] == pytest.approx(0.0105)
    assert "BTC-30OCT26-65000-C" in set(s1.symbol)
    assert s1.bid_price[list(s1.symbol).index("BTC-30OCT26-60000-C")] == pytest.approx(0.0110)
    assert s0.time_us == t0 and s1.time_us == t1
    # Early stop: far less than the whole stream was read.
    assert opener.responses[0].bytes_read < len(data) / 2
    assert meta["sha256"]["snap_t0.csv.gz"] == file_sha256(d / "snap_t0.csv.gz")
    assert meta["x_md5"] == "abc" and meta["bytes_streamed"] == opener.responses[0].bytes_read
    # Re-extraction gives byte-identical snapshots.
    cfg2 = _cfg(tmp_path / "again")
    meta2 = tardis.extract_month(cfg2, MONTH, opener=FakeOpener([data]))
    assert meta2["sha256"] == meta["sha256"]


def test_404_marks_unavailable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    meta = tardis.extract_month(cfg, MONTH, opener=FakeOpener([_http_error(404)]))
    assert meta["status"] == "unavailable"
    assert tardis.month_status(cfg, MONTH) == "unavailable"


def test_429_writes_quota_and_stops_session(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, months=("2026-08", "2026-09", "2026-10"))
    opener = FakeOpener([_http_error(429, "86100")])
    logs: list[str] = []
    summary = tardis.run_session(cfg, opener=opener, now=lambda: 1000.0, log=logs.append)
    assert len(opener.calls) == 1 and "locked" in summary["stopped"]
    quota = json.loads((cfg.cache_dir / "_quota.json").read_text())
    assert quota["not_before"] == pytest.approx(87100.0)
    # A new session before not_before makes no request at all.
    opener2 = FakeOpener([])
    s2 = tardis.run_session(cfg, opener=opener2, now=lambda: 2000.0, log=logs.append)
    assert opener2.calls == [] and "locked" in s2["stopped"]
    with pytest.raises(tardis.QuotaLocked):
        tardis.extract_month(cfg, "2026-08", opener=opener2, now=lambda: 2000.0)


def test_network_error_retries_then_succeeds(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    t0, t1 = cfg.times_us(MONTH)
    sleeps: list[float] = []
    opener = FakeOpener([TimeoutError("timed out"), _file(t0, t1)])
    meta = tardis.extract_month(cfg, MONTH, opener=opener, sleep=sleeps.append)
    assert meta["status"] == "ok" and meta["attempt"] == 2 and sleeps == [60.0]


def test_network_error_gives_up_after_three(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opener = FakeOpener([OSError("reset")] * 3)
    meta = tardis.extract_month(cfg, MONTH, opener=opener, sleep=lambda s: None)
    assert meta["status"] == "error" and len(opener.calls) == 3


@pytest.mark.parametrize("bad", ["columns", "order", "number"])
def test_invalid_stream_marked_invalid(tmp_path: Path, bad: str) -> None:
    cfg = _cfg(tmp_path)
    t0, t1 = cfg.times_us(MONTH)
    meta = tardis.extract_month(cfg, MONTH, opener=FakeOpener([_file(t0, t1, bad=bad)]))
    assert meta["status"] == "invalid", meta
    assert not (cfg.month_dir(MONTH) / "snap_t0.csv.gz").exists()


def test_truncated_stream_is_invalid(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    t0, t1 = cfg.times_us(MONTH)
    meta = tardis.extract_month(cfg, MONTH, opener=FakeOpener([_file(t0, t1, extra_minutes=0)]))
    assert meta["status"] == "invalid" and "before t1" in meta["error"]


def test_cache_cap_refuses(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, cache_cap_gb=1e-9)
    cfg.cache_dir.mkdir(parents=True)
    (cfg.cache_dir / "junk.bin").write_bytes(b"x" * 10)
    with pytest.raises(RuntimeError, match="cache"):
        tardis.extract_month(cfg, MONTH, opener=FakeOpener([]))


def test_session_order_cap_and_resume(tmp_path: Path) -> None:
    months = tuple(f"2025-{m:02d}" for m in range(1, 7))
    cfg = _cfg(tmp_path, months=months, max_requests_per_session=2)
    order = tardis.fetch_order(cfg)
    assert sorted(order) == list(months) and order != list(months)
    opener = FakeOpener([_http_error(404), _http_error(404)])
    s = tardis.run_session(cfg, opener=opener, log=lambda m: None, sleep=lambda x: None)
    assert [r["month"] for r in s["requested"]] == order[:2]
    assert "cap" in s["stopped"]
    opener2 = FakeOpener([_http_error(404)] * 4)
    s2 = tardis.run_session(cfg, opener=opener2, log=lambda m: None, sleep=lambda x: None)
    assert [r["month"] for r in s2["requested"]] == order[2:4]


def test_from_study_config() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "study_quick.toml").open("rb") as fh:
        study = tomllib.load(fh)
    cfg = tardis.TardisConfig.from_study(study)
    t0, t1 = cfg.times_us("2026-10")
    assert t1 - t0 == 15 * 60 * 1_000_000
    assert tardis._url("2026-10").endswith("/2026/10/01/OPTIONS.csv.gz")


# ------------------------------------------------------------------------- Deribit
def test_parse_instrument() -> None:
    und, exp, strike, is_call = deribit.parse_instrument("BTC-27NOV26-73000-P")
    assert (und, strike, is_call) == ("BTC", 73000.0, False)
    assert exp == 1795766400 * 1_000_000  # 2026-11-27 08:00 UTC
    assert deribit.parse_instrument("ETH-5DEC26-2d5-C")[2] == 2.5
    assert deribit.parse_instrument("BTC-PERPETUAL") is None
    assert deribit.parse_instrument("BTC_USDC-27NOV26-73000-P") is None


def test_snapshot_from_summaries_and_live_pair(tmp_path: Path) -> None:
    rows = [
        {
            "instrument_name": "BTC-27NOV26-73000-P",
            "bid_price": 0.05,
            "ask_price": 0.052,
            "mark_price": 0.051,
            "mark_iv": 48.2,
            "underlying_price": 70000.0,
            "creation_timestamp": 1_790_000_000_000,
        },
        {"instrument_name": "BTC-27NOV26-80000-C", "bid_price": None, "ask_price": 0.01},
        {"instrument_name": "BTC-PERPETUAL"},
    ]
    snap = deribit.snapshot_from_summaries(1_790_000_000_000_000, rows)
    assert len(snap) == 2 and snap.timestamp_us[0] == 1_790_000_000_000_000

    class Resp(FakeResponse):
        pass

    body = json.dumps({"result": rows}).encode()

    def opener(req: Any, timeout: float = 0) -> FakeResponse:
        assert "get_book_summary_by_currency" in req.full_url
        return Resp(body)

    sleeps: list[float] = []
    d = deribit.save_live_pair(
        tmp_path, horizon_min=15, currencies=("BTC",), opener=opener, sleep=sleeps.append
    )
    meta = json.loads((d / "meta.json").read_text())
    assert sleeps == [900.0] and meta["n_t0"] == 2 and "snap_t1.csv.gz" in meta["sha256"]
    with pytest.raises(ValueError):
        deribit.fetch_summary("SOL", opener=opener)
