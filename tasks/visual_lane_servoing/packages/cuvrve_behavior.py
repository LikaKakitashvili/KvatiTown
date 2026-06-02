from typing import List, Tuple
import numpy as np


def detect_curve(yellow_xs: List[int],white_xs:  List[int],curve_threshold: int = 350,
    ) -> Tuple[bool, int]:
    # Hint: xs[0] is the position closest to the robot, xs[-1] is farther ahead.
    # If the line shifts by more than curve_threshold pixels between near and far,
    # the road is curving. The sign of the shift tells you which way.
    shift = 0
    if len(yellow_xs) >= 2:
        shift = yellow_xs[-1] - yellow_xs[0]
    elif len(white_xs) >= 2:
        shift = white_xs[-1] - white_xs[0]

    if abs(shift) > curve_threshold:
        return True, int(np.sign(shift))

    return False, 0
