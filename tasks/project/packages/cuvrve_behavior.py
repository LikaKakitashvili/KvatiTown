from typing import List, Tuple
import numpy as np


def detect_curve(
    yellow_xs: List[int],
    white_xs:  List[int],
    curve_threshold: int = 350,
) -> Tuple[bool, int]:
    # Hint: xs[0] is the position closest to the robot, xs[-1] is farther ahead.
    # If the line shifts by more than curve_threshold pixels between near and far,
    # the road is curving. The sign of the shift tells you which way.
    return False, 0
