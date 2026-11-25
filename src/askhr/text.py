def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in the given text by replacing multiple spaces with a single space
    and removing leading/trailing whitespace.

    Args:
        text: The input text to normalize.

    Returns:
        str: The normalized text.
    """
    return " ".join(text.split())
