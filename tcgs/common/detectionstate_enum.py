from enum import Enum, auto

class DetectionState(Enum):
    RUNNING = auto()
    IDENTIFIED = auto()
    AMBIGUOUS = auto()
    ERRONEOUS = auto()

    def get_color_gbr(self):
        match self:
            case DetectionState.RUNNING: return (0, 0, 0)
            case DetectionState.IDENTIFIED: return (0, 200, 0)
            case DetectionState.AMBIGUOUS: return (0, 200, 200)
            case DetectionState.ERRONEOUS: return (0, 0, 200)
            case _: raise ValueError(f"Unknown State: {self}")