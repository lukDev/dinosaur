class Data:
    # --- Controller Parameters ---
    players = []
    obstacles = []
    gen = 0
    overall_time = 0.
    closest_obstacle = None
    second_closest_obstacle = None

    # --- Game Parameters ---
    field_size_x = 1200.
    field_size_y = 500.
    player_x = 100.
    baseline = 60.
    player_size = (80., 86.)
    activation_jump_threshold = 0.75
    obstacle_speed = 350.
    obstacle_start_x = field_size_x + 10.
    obstacle_sizes = [((64., 66.), .6), ((46., 90.), .4)]
    obstacle_time_dif_min = .85
    obstacle_time_dif_max = 1.6
    jump_speed_up = 500.
    gravity_acc = 1500.
    no_gravity_time_long = 0.18
    no_gravity_time_short = 0.085

    # --- NEAT Parameters ---
    population_size = 170
    disjoint_weight = 1.
    excess_weight = 1.
    weight_dif_weight = 0.4
    speciation_threshold = 0.7
    prob_crossover = 0.75
    prob_weight_mut = 0.8
    prob_weight_nudged = 0.9
    weight_nudge_limit = 0.1
    prob_new_neuron = 0.03
    prob_new_gene = 0.05
    current_innovation_number = 0

    # --- Methods ---
    @staticmethod
    def standard_obstacle_size_y():
        (size_y, _) = Data.obstacle_sizes[0][0]
        return size_y
