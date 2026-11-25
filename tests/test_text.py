import pytest

from askhr.text import normalize_whitespace


@pytest.mark.parametrize(
    ("input_text", "expected_output"),
    [
        ("  several   spaces  here  ", "several spaces here"),
        ("tabs\tand\nnewlines", "tabs and newlines"),
        ("  leading and trailing spaces  ", "leading and trailing spaces"),
        ("", ""),
        ("  \n\t  ", ""),
        ("  Congé   de  maternité ", "Congé de maternité"),
        ("Durée\u00a0:  12 mois", "Durée : 12 mois"),
    ],
)
def test_normalize_whitespace(input_text: str, expected_output: str) -> None:
    assert normalize_whitespace(input_text) == expected_output
