from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from biotradeai.data.catalysts import CatalystFileError, load_catalysts, upcoming

HEADER = "ticker,drug,event,date,source_url\n"


def _write(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    p = tmp_path / "catalysts.csv"
    p.write_text(header + body, encoding="utf-8")
    return p


def test_repo_file_loads() -> None:
    assert isinstance(load_catalysts(), list)


def test_valid_rows_and_upcoming(tmp_path: Path) -> None:
    p = _write(tmp_path,
               "ONCO,ONC-101,pdufa,2026-06-15,https://example.org/pr1\n"
               "\n"
               "DILU,DLT-7,PHASE3_READOUT,2025-01-10,https://example.org/pr2\n")
    rows = load_catalysts(p)
    assert [r.event for r in rows] == ["PDUFA", "PHASE3_READOUT"]
    assert [r.ticker for r in upcoming(rows, today=date(2026, 1, 1))] == ["ONCO"]


@pytest.mark.parametrize(
    ("row", "msg"),
    [
        ("onco,X,PDUFA,2026-01-01,https://e.org\n", "ticker"),
        ("ONCO,,PDUFA,2026-01-01,https://e.org\n", "drug"),
        ("ONCO,X,EARNINGS,2026-01-01,https://e.org\n", "event"),
        ("ONCO,X,PDUFA,01/02/2026,https://e.org\n", "date"),
        ("ONCO,X,PDUFA,2026-01-01,ftp://e.org\n", "source_url"),
        ("ONCO,X,PDUFA,2026-01-01,https://e.org\n" * 2, "duplicate"),
    ],
)
def test_bad_rows(tmp_path: Path, row: str, msg: str) -> None:
    with pytest.raises(CatalystFileError, match=msg):
        load_catalysts(_write(tmp_path, row))


def test_bad_header(tmp_path: Path) -> None:
    with pytest.raises(CatalystFileError, match="header"):
        load_catalysts(_write(tmp_path, "", header="ticker,drug,date\n"))
