nodes_possibilities = [16,32,64] # List of the possible number of nodes per layer

momentum_bnd = {
    "start": 0,
    "end": 1,
    "step" : 0.1
} # dictionnary containing the momentum boundaries

learning_rate_bnd = {
    "start_pow" : -4,
    "end_pow" : 1
}

random_grid_search_iter = 10

number_of_hidden_layers = 3
