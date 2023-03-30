from neural import * 
from optimization import *
from run_function import *
import config as config
 
if __name__ == '__main__':
    # Load the dataset
    (X_train_flattened, X_test_flattened), (y_train, y_test) = loadDataset(keras.datasets.mnist.load_data())
    #(X_train_flattened, X_test_flattened), (y_train, y_test) = loadDataset(keras.datasets.fashion_mnist.load_data())
    data = [X_train_flattened, y_train, X_test_flattened, y_test]
    
    # # Optimize hyperparameters with grid search
    # training_results = grid_search(data, y_test, random=False)
    # training_results.to_csv('grid_search_results.csv', sep=";")
    
    # Optimize hyperparameters with random grid search
    training_results = grid_search(data, random=True, number_of_random_choice=config.random_grid_search_iter)
    training_results.to_csv(f'results/random_grid_search_results.csv', sep=";")
    
    # Optimize hyperparameters with bayesian technique

    # search_space = {
    #     'learning_rate' : [-5,0] ,
    #     'momentum' : [0,1] ,
    #     'num_hidden_layers' : [3,3] ,
    #     'num_neurons_per_layer' : [2,6] ,
    # }

    # run_bayesian(data, search_space, "cont_fixed_start_ei_2_starts_2", acquisition_func="ei", num_trials=10, iterations_per_trial=8)
    



