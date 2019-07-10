import math
import random
from data import Data


def sigmoid_activation(x):
        return 1. / (1. + math.exp(-4.9*x))


def sigmoid_weight(x):
        return 2. / (1. + math.exp(-4.9 * x)) - 1.


def distance(p1x, p1y, p2x, p2y):
        return math.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)


def center(x, y, size_x, size_y):
        return (x + size_x / 2, y + size_y / 2)


def random_decision(probability):
        return random.uniform(0., 1.) <= probability


def random_range_decision(possibilities):
        current_range_start = 0.
        selection = random.uniform(0., 1.)
        for (pos, prob) in possibilities:
                if selection >= current_range_start and selection < current_range_start + prob:
                        return pos
                else:
                        current_range_start += prob


def are_rects_intersecting(x1, y1, w1, h1, x2, y2, w2, h2):
        return ((x1 >= x2 and x1 <= x2 + w2) or (x2 >= x1 and x2 <= x1 + w1)) and ((y1 >= y2 and y1 <= y2 + h2) or (y2 >= y1 and y2 <= y1 + h1))


def x_dif_to_activation(x1, x2):
        return sigmoid_activation(float(abs(x1 - x2)) / float(Data.field_size_x))


def obstacle_size_y_to_activation(size_y):
        heights = [h for (h, _) in Data.obstacle_sizes_y]
        max_height = max(heights)
        min_height = min(heights)
        return (size_y - min_height) / (max_height - min_height)
