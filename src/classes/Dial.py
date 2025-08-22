

from enum import Enum

class Action(str, Enum):
    """Enum class for dial pad keys."""
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"
    STAR = '*'
    HASH = '#'
    NONE = None