"""Unit tests for the 192 matching rule (answer_text + mixed alphabet + date
granularity). Run on Kaggle with pymorphy3; the stemmer fallback in a bare
sandbox gives identical outcomes on these cases.
"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "match_192", Path(__file__).resolve().parents[1] / "src" / "match_192.py")
m192 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m192)

correct = m192.correct_v2


# mixed alphabet (Latin homoglyphs inside Cyrillic answers)
@pytest.mark.parametrize("ans", ["Тennis", "теннис", "Бonn", "бонн", "Нicotиновая"])
def test_homoglyph_cyrillic(ans):
    refs = ["теннис", "Бонн", "Никотиновая"]
    assert any(correct(ans, [r], "entity") for r in refs)


def test_homoglyph_c_to_k():
    assert correct("Нicotиновая", ["Никотиновая"], "entity")


# RuBQ answer_text as an additional reference
def test_answer_text_reference():
    assert correct("Юрий Гагарин", ["Юрий Алексеевич Гагарин", "Юрий Гагарин"], "entity")


# date granularity (D-028: less precise = error in the main variant)
@pytest.mark.parametrize("ans", ["IX век", "860 год примерно", "IX"])
def test_century_is_error_main(ans):
    assert not correct(ans, ["0860-01-01T00:00:00Z"], "date")


def test_century_granted_soft():
    assert correct("IX век", ["0860-01-01T00:00:00Z"], "date", soft=True)


def test_wrong_day_is_error():
    assert not correct("1991-12-08", ["1991-12-26T00:00:00Z"], "date")


def test_bare_year_matches_placeholder():
    assert correct("860", ["0860-01-01T00:00:00Z"], "date")
    assert correct("1842", ["1842-01-01T00:00:00Z"], "date")


def test_year_only_answer_to_full_ref_is_error_main():
    assert not correct("1991", ["1991-12-26T00:00:00Z"], "date")


def test_year_only_answer_to_full_ref_granted_soft():
    assert correct("1991", ["1991-12-26T00:00:00Z"], "date", soft=True)


def test_wrong_day_is_error_soft():
    assert not correct("1991-12-08", ["1991-12-26T00:00:00Z"], "date", soft=True)


# numbers: rounding tolerance for quantities, strict for bare years
def test_quantity_rounding_soft():
    assert not correct("237", ["236"], "number")
    assert correct("237 Ватт", ["236"], "number", soft=True)
    assert correct("207600平方公里", ["207600.496983"], "number", soft=True)


@pytest.mark.parametrize("ans,ref", [("1918", "1917"), ("1993", "1992"), ("1783", "1784")])
def test_off_by_one_year_not_granted(ans, ref):
    assert not correct(ans, [ref], "number")
    assert not correct(ans, [ref], "number", soft=True)


def test_exact_number():
    assert correct("236", ["236"], "number")


# containment stays a soft-only feature
def test_subset_answer_is_error_main():
    assert not correct("Ислам", ["ханафизм суннитского толка"], "entity")
