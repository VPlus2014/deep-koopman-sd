import math
import numpy as np

import sys
from pathlib import Path

__DIR_WS = Path(__file__).resolve().parent.parent

sys.path.append(str(__DIR_WS))
from data.dynamics.mathext import *


_PI = math.pi
_2PI = 2 * _PI
_PI_HALF = _PI * 0.5
