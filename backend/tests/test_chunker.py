"""Kapitel-Erkennung im Chunker (Kapitel-Wort vs. Großbuchstaben-Überschriften)."""

from __future__ import annotations

from app.ingestion.chunker import chapter_starts, chunk_book

BODY = "Ein ganz normaler Absatz mit Text, der weitergeht."


def test_keyword_headings_take_precedence_over_caps_subtitles():
    text = "\n\n".join(["CHAPTER I", "JONATHAN HARKER'S JOURNAL", BODY, "CHAPTER II", BODY])
    assert [t for _, t in chapter_starts(text)] == ["CHAPTER I", "CHAPTER II"]


def test_caps_headings_after_marker_or_running_footer():
    footer = "Page | 19 Harry Potter and the Goblet of Fire - J.K. Rowling"
    parts = ["/", "THE RIDDLE HOUSE", BODY, "I", "DUDLEY DEMENTED", BODY]
    for n in range(5):  # wiederkehrende Fußzeile mit wechselnder Seitenzahl
        parts += [BODY, footer.replace("19", str(20 + n))]
    parts += ["THE SCAR", BODY]
    text = "\n\n".join(parts)
    starts = chapter_starts(text)
    assert [t for _, t in starts] == ["THE RIDDLE HOUSE", "DUDLEY DEMENTED", "THE SCAR"]
    assert all(text[o:].startswith(t) for o, t in starts)


def test_caps_inside_prose_are_not_headings():
    # „WHAM“ nach Dialog, Schlagzeile nach einem Satz — kein Kapitel.
    text = "\n\n".join(["I", "BRAN", BODY, '"Dudley, shut — "', "WHAM", BODY,
                         "He read:", "GRINGOTTS BREAK-IN LATEST", BODY, "STOP!"])
    assert [t for _, t in chapter_starts(text)] == ["BRAN"]


def test_chunks_carry_caps_chapter():
    text = "\n\n".join(["I", "PROLOGUE", BODY, "II", "CATELYN", BODY])
    chunks = chunk_book(text, target_chars=10, overlap_chars=0)
    assert [c.chapter for c in chunks if BODY in c.text] == ["PROLOGUE", "CATELYN"]
