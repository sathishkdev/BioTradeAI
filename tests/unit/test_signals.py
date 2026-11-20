"""Deterministic filing-text signals: dates/windows, catalysts, drugs, outcomes, changes."""

from __future__ import annotations

from datetime import date

import pytest

from biotradeai.data.signals import (
    OutcomeKind,
    Precision,
    drug_mentions,
    find_catalysts,
    find_material_changes,
    find_outcomes,
    mentions_any,
    parse_when,
)


@pytest.mark.parametrize(("text", "start", "end", "prec"), [
    ("on March 15, 2027", date(2027, 3, 15), date(2027, 3, 15), Precision.DAY),
    ("by 2027-03-15", date(2027, 3, 15), date(2027, 3, 15), Precision.DAY),
    ("in the first quarter of 2027", date(2027, 1, 1), date(2027, 3, 31), Precision.QUARTER),
    ("in Q4 2026", date(2026, 10, 1), date(2026, 12, 31), Precision.QUARTER),
    ("in 2Q 2027", date(2027, 4, 1), date(2027, 6, 30), Precision.QUARTER),
    ("in the second half of 2027", date(2027, 7, 1), date(2027, 12, 31), Precision.HALF),
    ("in 1H 2027", date(2027, 1, 1), date(2027, 6, 30), Precision.HALF),
    ("in mid-2027", date(2027, 4, 1), date(2027, 9, 30), Precision.HALF),
    ("by year-end 2026", date(2026, 10, 1), date(2026, 12, 31), Precision.QUARTER),
    ("in February 2028", date(2028, 2, 1), date(2028, 2, 29), Precision.MONTH),
])
def test_parse_when(text: str, start: date, end: date, prec: Precision) -> None:
    w = parse_when(text)
    assert w is not None and (w.start, w.end, w.precision) == (start, end, prec)
    assert (w.exact is not None) is (prec is Precision.DAY)


def test_parse_when_none_and_invalid() -> None:
    assert parse_when("no date here") is None
    assert parse_when("on February 30, 2027") is None


def test_drug_mentions() -> None:
    s = "ABC-123 (abcitinib) and XY101 beat COVID-19 controls in Q1 2027 per NCT01234567."
    assert drug_mentions(s) == ["ABC-123", "XY101", "abcitinib"]
    assert mentions_any(("ABC123",), ("ABC-123", "abcitinib"))
    assert not mentions_any(("XY101",), ("ABC-123",))


def test_find_catalysts() -> None:
    text = ("Acme Bio today announced that the FDA accepted its NDA for ABC-123. "
            "The FDA assigned a PDUFA target action date of March 15, 2027. "
            "Topline data from the pivotal Phase 3 RESOLVE trial of XY-101 are expected "
            "in the first half of 2027. "
            "The company previously reported positive Phase 2 data in May 2026. "
            "An advisory committee meeting is planned for January 20, 2027.")
    got = {(c.event_type, c.when.precision, c.drugs[:1]) for c in find_catalysts(text)}
    assert ("PDUFA", Precision.DAY, ("ABC-123",)) in got  # drug from previous sentence
    assert ("PHASE3_READOUT", Precision.HALF, ("XY-101",)) in got
    assert ("ADCOM", Precision.DAY, ()) in got
    assert not any(c.event_type == "PHASE2_READOUT" for c in find_catalysts(text))  # past


def test_find_outcomes() -> None:
    text = ("The Phase 3 trial of ABC-123 met its primary endpoint. "
            "XY-101 did not meet its primary endpoint. "
            "The Company received a complete response letter for ZZ-901 from the FDA. "
            "The FDA approved Abcora (ABC-123) for adults with condition X. "
            "Abcora is now commercially available in the United States. "
            "The Company discontinued development of QQ-77. "
            "If the FDA issues a complete response letter, we could be delayed.")
    kinds = [(o.kind, o.drugs[:1]) for o in find_outcomes(text)]
    assert (OutcomeKind.POSITIVE, ("ABC-123",)) in kinds
    assert (OutcomeKind.NEGATIVE, ("XY-101",)) in kinds
    assert (OutcomeKind.CRL, ("ZZ-901",)) in kinds
    assert any(k is OutcomeKind.APPROVED for k, _ in kinds)
    assert any(k is OutcomeKind.LAUNCHED for k, _ in kinds)
    assert (OutcomeKind.DISCONTINUED, ("QQ-77",)) in kinds
    assert sum(k is OutcomeKind.CRL for k, _ in kinds) == 1  # hedged sentence ignored


def test_planned_launch_is_not_a_launch() -> None:
    assert not find_outcomes("We plan a commercial launch of Abcora in 2027.")


def test_find_material_changes() -> None:
    text = ("Acme entered into a definitive merger agreement with BigPharma. "
            "Acme signed an exclusive worldwide license agreement for ABC-123 with PartnerCo. "
            "Acme may consider strategic alternatives, including a merger agreement.")
    got = [(c.kind, c.drugs) for c in find_material_changes(text)]
    assert got == [("acquisition", ()), ("license", ("ABC-123",))]
