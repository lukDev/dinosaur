from utils.data import Data
from game.entity import Obstacle
import pyglet
import random
from utils import math_helper


class Controller:
    @staticmethod
    def update(dt):
        if dt > 0.5:
            dt = 0.5

        obstacles_ahead = [obs for obs in Data.obstacles if obs.x + obs.size_x >= Data.player_x]
        if len(obstacles_ahead) < 1:
            Data.closest_obstacle = None
        else:
            closest_obstacles = sorted(obstacles_ahead, key=lambda obs: obs.x)
            Data.closest_obstacle = closest_obstacles[0]
            if len(closest_obstacles) >= 2:
                Data.second_closest_obstacle = closest_obstacles[1]
            else:
                Data.second_closest_obstacle = None

        alive_count = 0
        for player in Data.players:
            player.move(dt)
            if player.alive:
                alive_count += 1

        obstacles_to_remove = []
        for obstacle in Data.obstacles:
            obstacle.move(dt)
            if obstacle.x <= -Data.obstacle_size_x:
                obstacles_to_remove.append(obstacle)
        new_obstacles = [obstacle for obstacle in Data.obstacles if obstacle not in obstacles_to_remove]
        Data.obstacles = new_obstacles

        return alive_count

    @staticmethod
    def send_obstacle(_):
        size_y = math_helper.random_range_decision(Data.obstacle_sizes_y)
        Data.obstacles.append(Obstacle(x=Data.obstacle_start_x, y=Data.baseline, size_x=Data.obstacle_size_x, size_y=size_y))
        pyglet.clock.schedule_once(Controller.send_obstacle, random.uniform(Data.obstacle_time_dif_min, Data.obstacle_time_dif_max))

    @staticmethod
    def reset_level():
        Data.obstacles = []
        pyglet.clock.unschedule(Controller.send_obstacle)
        Controller.send_obstacle(0)
