#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def create_bounds(min, max, dim):
    bounds = [(min, max) for _ in range(dim)]
    return np.array(bounds)


def print_color(str, color):
    print(f"\033[{color}m" + str + "\033[0m")


print_pink = lambda str: print_color(str, 95)
print_blue = lambda str: print_color(str, 94)
print_green = lambda str: print_color(str, 32)
print_purple = lambda str: print_color(str, 35)
print_bright_red = lambda str: print_color(str, 91)
