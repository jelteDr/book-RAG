"""Kapitel-Erkennung im Chunker (Kapitel-Wort vs. Großbuchstaben-Überschriften)."""

from __future__ import annotations

from app.ingestion.chunker import chapter_starts, chunk_book

BODY = "Ein ganz normaler Absatz mit Text, der weitergeht."
_n = 0


def body() -> str:
    """Erzähltext, jeder Absatz anders (identische Absätze ≥ 5× gelten als Kopf-/Fußzeile;
    Ziffern zählen dabei nicht — deshalb Buchstaben statt Nummern)."""
    global _n
    _n += 1
    word = "".join("abcdefghij"[int(d)] for d in str(_n))
    return f"{BODY} Absatz {word}."


def test_keyword_headings_take_precedence_over_caps_subtitles():
    text = "\n\n".join(["CHAPTER I", "JONATHAN HARKER'S JOURNAL", body(), "CHAPTER II", body(),
                         "CHAPTER III", body()])
    assert [t for _, t in chapter_starts(text)] == ["CHAPTER I", "CHAPTER II", "CHAPTER III"]


def test_caps_headings_after_marker_or_running_footer():
    footer = "Page | 19 Harry Potter and the Goblet of Fire - J.K. Rowling"
    parts = ["/", "THE RIDDLE HOUSE", body(), "I", "DUDLEY DEMENTED", body()]
    for n in range(5):  # wiederkehrende Fußzeile mit wechselnder Seitenzahl
        parts += [body(), footer.replace("19", str(20 + n))]
    parts += ["THE SCAR", body()]
    text = "\n\n".join(parts)
    starts = chapter_starts(text)
    assert [t for _, t in starts] == ["THE RIDDLE HOUSE", "DUDLEY DEMENTED", "THE SCAR"]
    assert all(text[o:].startswith(t) for o, t in starts)


def test_caps_inside_prose_are_not_headings():
    # „WHAM“ nach Dialog, Schlagzeile nach einem Satz — kein Kapitel.
    text = "\n\n".join(["I", "OWL POST", body(), '"Dudley, shut — "', "WHAM", body(),
                         "II", "THE KNIGHT BUS", body(), "He read:", "GRINGOTTS BREAK-IN LATEST",
                         body(), "III", "THE DEMENTOR", body(), "STOP!"])
    assert [t for _, t in chapter_starts(text)] == ["OWL POST", "THE KNIGHT BUS", "THE DEMENTOR"]


def test_chunks_carry_caps_chapter():
    text = "\n\n".join(["PROLOGUE", body(), "CATELYN", body(), "BRAN", body(), "CATELYN", body(),
                         "BRAN", body()])
    chunks = chunk_book(text, target_chars=10, overlap_chars=0)
    assert [c.chapter for c in chunks if BODY in c.text] == [
        "PROLOGUE", "CATELYN", "BRAN", "CATELYN", "BRAN"]


def _book(*chapters: str, filler: int = 3) -> str:
    parts = []
    for title in chapters:
        parts += [title] + [body() for _ in range(filler)]
    return "\n\n".join(parts)


def test_repeated_pov_names_are_chapters_unique_caps_are_not():
    # GoT-Muster: POV-Namen wiederholen sich, Anhang/OCR-Kopfzeilen sind einmalig.
    text = _book("PROLOGUE", "BRAN", "CATELYN", "BRAN", "CATELYN", "HOUSE STARK", "A GAML OF THRONES 109")
    assert [t for _, t in chapter_starts(text)] == ["PROLOGUE", "BRAN", "CATELYN", "BRAN", "CATELYN"]


def test_single_keyword_line_does_not_force_keyword_mode():
    text = _book("Book Four: A Song of Ice and Fire", "CERSEI", "JAIME", "CERSEI", "JAIME")
    assert [t for _, t in chapter_starts(text)] == ["CERSEI", "JAIME", "CERSEI", "JAIME"]


def test_numbered_toc_at_start_is_not_taken_as_chapters():
    # Inhaltsverzeichnis „1 / JON / 2 / BRAN …“ am Anfang, echte Kapitel danach ohne Marker.
    toc = []
    for n, name in enumerate(["JON", "BRAN", "TYRION"], start=1):
        toc += [str(n), name]
    body = _book("JON", "BRAN", "TYRION", "JON", filler=30)
    text = "\n\n".join(toc) + "\n\n" + body
    titles = [t for o, t in chapter_starts(text) if o > len("\n\n".join(toc))]
    assert titles == ["JON", "BRAN", "TYRION", "JON"]
