from enum import Enum

class ActionType(str, Enum):
    wait = ">wait for input"
    # restart = "restart call"
    press = ">press [number]"