from __future__ import annotations

from oneehr.features.names import display_name, display_names


def test_display_name_numeric():
    assert display_name("num__Age") == "Age"


def test_display_name_categorical():
    assert display_name("cat__Sex__M") == "Sex=M"


def test_display_names_preserves_order():
    assert display_names(["num__Age", "cat__Sex__M"]) == ["Age", "Sex=M"]

