class Data:
    # --- controller parameters ---
    players = []
    obstacles = []
    gen = 0
    overall_time = 0.
    closest_obstacle = None
    second_closest_obstacle = None

    # --- game parameters ---
    field_size_x = 800.
    field_size_y = 300.
    player_x = 100.
    baseline = 60.
    player_size_x = 20.
    player_size_y = 50.
    activation_jump_threshold = 0.75
    obstacle_speed = 400.
    obstacle_start_x = field_size_x + 10.
    obstacle_size_x = 20.
    obstacle_sizes_y = [(40., .6), (60., .4)]
    obstacle_time_dif_min = 0.45
    obstacle_time_dif_max = 1.5
    large_obstacle_min_time_dif = 0.7
    jump_speed_up = 300.
    gravity_acc = 1500.
    no_gravity_time_long = 0.18
    no_gravity_time_short = 0.085

    # --- NEAT parameters ---
    population_size = 170
    disjoint_weight = 1.
    excess_weight = 1.
    weight_dif_weight = 0.4
    speciation_threshold = 0.7
    plateau_limit = 15
    prob_crossover = 0.75
    prob_weight_mut = 0.8
    prob_weight_nudged = 0.9
    weight_nudge_limit = 0.1
    prob_new_neuron = 0.03
    prob_new_gene = 0.05
    current_innovation_number = 0

    # --- methods ---
    @staticmethod
    def standard_obstacle_size_y():
        (size_y, _) = Data.obstacle_sizes_y[0]
        return size_y
