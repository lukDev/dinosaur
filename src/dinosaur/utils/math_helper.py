import math
import random
from dinosaur.utils.data import Data


# --- 2D Maths ---

# returns the distance between two points in a plane, defined by two coordinates each
def distance(p1x, p1y, p2x, p2y):
    return math.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)


# returns the center coordinates of a rectangle defined by its coordinates and size
def center(x, y, size_x, size_y):
    return x + size_x / 2, y + size_y / 2


# checks if two rectangles, defined by their positions and sizes, are intersecting
def are_rects_intersecting(x1, y1, w1, h1, x2, y2, w2, h2):
    return ((x2 <= x1 <= x2 + w2) or (x1 <= x2 <= x1 + w1)) and ((y2 <= y1 <= y2 + h2) or (y1 <= y2 <= y1 + h1))


# --- Random Decisions ---

# decides yes with a given probability p, no with 1 - p
def random_decision(probability):
    return random.uniform(0., 1.) <= probability


# decides which of the given options will be selected, based on their probabilities
# the sum of all probabilities must be 1
def random_range_decision(possibilities):
    current_range_start = 0.
    selection = random.uniform(0., 1.)
    for (pos, prob) in possibilities:
        if current_range_start <= selection < current_range_start + prob:
            return pos
        else:
            current_range_start += prob


# --- Activations ---

# returns the result of a scaled sigmoid function for a given x
def sigmoid_activation(x):
    return 1. / (1. + math.exp(-4.9*x))


# returns the result of a scaled and shifted sigmoid function for a given x
def sigmoid_weight(x):
    return 2. / (1. + math.exp(-4.9 * x)) - 1.


# returns the difference between two x components in relation to the total available x value range,
# modified by the sigmoid function to make the value appropriate for an activation
def x_dif_to_activation(x1, x2):
    return sigmoid_activation(float(abs(x1 - x2)) / float(Data.field_size_x))


# returns the height of an obstacle in relation to the total available height range, modified by the sigmoid function
def obstacle_size_y_to_activation(size_y):
    heights = [h for ((_, h), _) in Data.obstacle_sizes]
    max_height = max(heights)
    min_height = min(heights)
    return (size_y - min_height) / (max_height - min_height)
