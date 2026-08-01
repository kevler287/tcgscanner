from enum import Enum, auto

class DetectionState(Enum):
    IDENTIFIED = auto()
    AMBIGUOUS = auto()
    ERRONEOUS = auto()

    def get_color(self):
        match self:
            case DetectionState.IDENTIFIED: return (0, 200, 0)
            case DetectionState.AMBIGUOUS: return (200, 200, 0)
            case DetectionState.ERRONEOUS: return (200, 0, 0)
            case _: raise ValueError(f"Unknown State: {self}")