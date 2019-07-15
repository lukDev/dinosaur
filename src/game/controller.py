from utils.data import Data
import game.entity as entity
import pyglet
import random
from utils import math_helper
from neat.network import Network


class Controller:
    # updates and creates new obstacles, triggers the players' movement
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
            if obstacle.x <= -obstacle.size_x:
                obstacles_to_remove.append(obstacle)
        new_obstacles = [obstacle for obstacle in Data.obstacles if obstacle not in obstacles_to_remove]
        Data.obstacles = new_obstacles

        return alive_count

    # sends a new obstacle towards the player and sets a timer for the next one
    @staticmethod
    def send_obstacle(_):
        size = math_helper.random_range_decision(Data.obstacle_sizes)
        Data.obstacles.append(entity.Obstacle(x=Data.obstacle_start_x, y=Data.baseline, size_x=size[0], size_y=size[1]))
        pyglet.clock.schedule_once(Controller.send_obstacle, random.uniform(Data.obstacle_time_dif_min, Data.obstacle_time_dif_max))

    # resets the level
    @staticmethod
    def reset_level():
        Data.obstacles = []
        pyglet.clock.unschedule(Controller.send_obstacle)
        Controller.send_obstacle(0)

    # creates the first generation's players, giving them a neural network with just input and output neurons,
    # but without any genes
    @staticmethod
    def init_starter_players():
        Data.players = []
        Data.current_innovation_number = 0  # setting the innovation number so that it's valid after initialization

        for _ in range(Data.population_size):
            genes = []
            add_ins = [0, 1, 2]
            add_outs = [3, 4]

            network = Network(genes, add_ins, add_outs)
            Data.players.append(entity.Player(network, x=Data.player_x, y=Data.baseline, size_x=Data.player_size[0],
                                              size_y=Data.player_size[1]))

    # creates new players from a given list of neural networks
    @staticmethod
    def create_new_players(nets):
        Data.players = []

        for net in nets:
            Data.players.append(entity.Player(network=net, x=Data.player_x, y=Data.baseline, size_x=Data.player_size[0],
                                              size_y=Data.player_size[1]))
