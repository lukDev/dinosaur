import pyglet
from pyglet.gl import GL_QUADS
from data import Data
import math_helper


class Entity:
    def __init__(self, x, y, size_x, size_y):
        self.x = x
        self.y = y
        self.size_x = size_x
        self.size_y = size_y
        self.time_since_jump = 0
        self.long_jump = False

    def render(self):
        coords = [self.x, self.y, self.x + self.size_x, self.y, self.x + self.size_x, self.y + self.size_y, self.x, self.y + self.size_y]
        pyglet.graphics.draw(4, GL_QUADS, ("v2f", coords))

    def center(self):
        return math_helper.center(self.x, self.y, self.size_x, self.size_y)


class Player(Entity):
    def __init__(self, network, x=50., y=50., size_x=20., size_y=60.):
        Entity.__init__(self, x, y, size_x, size_y)
        self.network = network
        self.alive = True
        self.fitness = 0

    def move(self, dt):
        if not self.alive:
            return

        if self.y < Data.baseline:
            self.y = Data.baseline

        obs_1 = Data.closest_obstacle
        obs_2 = Data.second_closest_obstacle

        should_jump_long = False
        should_jump_short = False
        if obs_1 is not None and obs_2 is not None:
            in_activations = [
                1.,                                                         # bias
                math_helper.x_dif_to_activation(self.x, obs_1.x),           # dif to closest obstacle
                math_helper.x_dif_to_activation(self.x, obs_2.x),           # dif to second closest obstacle
                math_helper.obstacle_size_y_to_activation(obs_1.size_y)     # obstacle height
            ]
            out_activations = self.network.get_activations(in_activations)
            if len(out_activations) == 2:
                if out_activations[0] > Data.activation_jump_threshold:
                    if out_activations[1] > Data.activation_jump_threshold:
                        should_jump_long = out_activations[0] > out_activations[1]
                        should_jump_short = not should_jump_long
                    else:
                        should_jump_long = True
                else:
                    should_jump_short = out_activations[1] > Data.activation_jump_threshold
            else:
                print("unexpected number of out activations")

        if (should_jump_long or should_jump_short) and self.y == Data.baseline:
            self.time_since_jump = 0.
            self.long_jump = should_jump_long
            self.y += .1  # to activate the jump calculations

        if self.y > Data.baseline:
            self.time_since_jump += dt
            up_time = self.time_since_jump
            down_time = max(0., self.time_since_jump - (Data.no_gravity_time_long if self.long_jump else Data.no_gravity_time_short))
            self.y = Data.baseline + max(0, Data.jump_speed_up * up_time - 0.5 * Data.gravity_acc * down_time**2)

        self.fitness += dt

        if obs_1 is not None:
            self.alive = not math_helper.are_rects_intersecting(self.x, self.y, self.size_x, self.size_y, obs_1.x, obs_1.y, obs_1.size_x, obs_1.size_y)


class Obstacle(Entity):
    def __init__(self, x, y, size_x=20., size_y=40.):
        Entity.__init__(self, x, y, size_x, size_y)

    def move(self, dt):
        self.x -= dt * Data.obstacle_speed
