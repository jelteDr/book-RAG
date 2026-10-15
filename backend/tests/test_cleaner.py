"""Cleaner: Absatzerkennung in hart umbrochenen Texten ohne Leerzeilen (GoT-Dateien)."""

from __future__ import annotations

from app.ingestion.cleaner import clean

FULL = "x" * 95  # volle Zeile ohne Satzende


def _wrapped(*paragraph_lines: list[str]) -> str:
    return "\n".join(line for para in paragraph_lines for line in para)


def test_blank_line_texts_keep_old_behaviour():
    text = "Erste Zeile\nzweite Zeile.\n\nNeuer Absatz hier.\n\n"
    cleaned, report = clean(text)
    assert cleaned == "Erste Zeile zweite Zeile.\n\nNeuer Absatz hier."
    assert report.paragraphs_by_line_length is False


def test_hard_wrapped_text_splits_on_short_line_with_sentence_end():
    text = _wrapped(
        [FULL, FULL, "was concerned."],
        [FULL, "all shared the laugh."],
        ["BRAN"],
        [FULL, "beheaded."],
    )
    cleaned, report = clean(text)
    assert report.paragraphs_by_line_length is True
    paras = cleaned.split("\n\n")
    assert len(paras) == 4
    assert paras[0].endswith("was concerned.") and paras[2] == "BRAN"


def test_page_breaks_inside_sentence_are_not_paragraph_ends():
    lines = [FULL] * 40 + ["half-bored, half-distracted", "Page 2", "", "long enough to stop."]
    cleaned, report = clean("\n".join(lines))
    assert report.paragraphs_by_line_length is True
    assert "\n\n" not in cleaned and "Page 2" not in cleaned
    assert "half-distracted long enough" in cleaned
