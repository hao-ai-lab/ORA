"""Reading the Native provider admission probe's image answer."""
from __future__ import annotations

import re
from typing import Any

#: The probe image is solid red. An answer naming a red-family colour word is a
#: correct reading, also inside a compound (providers answered "vermilion" and
#: "orange-red" and were refused by an exact-word check).
RED_IMAGE_WORDS = frozenset({"red", "reddish", "crimson", "scarlet", "vermilion", "vermillion"})


def image_answer_word(content: Any) -> str:
    """The probe answer lower-cased without surrounding space or punctuation."""
    return content.strip().strip(".!,;:'\"").strip().lower() if isinstance(content, str) else ""


def names_red(answer: str) -> bool:
    return any(word in RED_IMAGE_WORDS for word in re.findall(r"[a-z]+", answer))
