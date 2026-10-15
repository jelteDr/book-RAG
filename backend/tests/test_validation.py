"""Validierung von Buch-Titel/-Autor (PATCH /books/{id})."""

from app.validation import validate_book_field


def test_title_required_and_trimmed():
    assert validate_book_field("  ", "Titel", required=True) == ["Titel darf nicht leer sein."]
    assert validate_book_field(" A Storm of Swords ", "Titel", required=True) == []


def test_author_optional_but_checked():
    assert validate_book_field("", "Autor", required=False) == []
    assert validate_book_field("x" * 201, "Autor", required=False)
    assert validate_book_field("J. K.\x07 Rowling", "Autor", required=False)
