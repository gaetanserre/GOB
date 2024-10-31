#
# Created in 2024 by Gaëtan Serré
#

from enum import Enum
from functools import total_ordering

@total_ordering
class Difficulty(Enum):
  Easy = 5
  Medium = 15
  Hard = 50
  Very_Hard = 100

  def __lt__(self, other):
    return self.value <= other.value