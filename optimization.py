import time as time 
import math
import numpy as np
import pandas as pd
from neural import *
import config as config

from random import randint, uniform
from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

def generate_hyperparameters_grid():
    """Generate all the hyperparameters combinations

    Returns:
        list: List of all hyperparameters possibilities
    """
    
    momentum = [x for x in np.arange(config.momentum_bnd["start"], config.momentum_bnd["end"], config.momentum_bnd["step"])]
    learning_rate = [10**x for x in range (config.learning_rate_bnd["start_pow"],config.learning_rate_bnd["end_pow"])]
    number_of_hidden_layers = config.number_of_hidden_layers
    number_of_nodes_per_layer = [list(possibility) + [10] for possibility in list(product(*[config.nodes_possibilities for i in range(number_of_hidden_layers)]))]

    return list(product(learning_rate,momentum, [number_of_hidden_layers], number_of_nodes_per_layer)) # generate all the possibilities



def grid_search(data, random = False, number_of_random_choice = 5):
    """Optimize the hyperparameters with a grid search

    Args:
        data (list) : List containing the data extracted from the dataset
        random (bool, optional): Boolean to say if we want to perform a random grid search or a grid search. Defaults to False.
        number_of_random_choice (int, optional): Number of random combination to pick if it's a random grid search. Defaults to 5.

    Returns:
        Dataframe: Dataframe containing the results of the (random) grid search
    """
    X_train_flattened = data[0]
    y_train = data[1]
    X_test_flattened = data[2]
    y_test = data[3]
    
    grid = generate_hyperparameters_grid()
    results = {
        'Learning_rate' : [],
        'Momentum' : [],
        'Number_of_hidden_layers' : [],
        'Number_of_nodes_per_layer' : [],
        'Time' : [],
        'training_accuracies' : [],
        'training_losses' : [],
        'test_loss' : [],
        'test_acc' : [],
        'max_acc' : [],
    }
    
    evaluations_number = number_of_random_choice if random else len(grid)
    for i in range(evaluations_number):
        index = randint(0, len(grid)-1) if random else i 
        neural_network = create_neural(
            learning_rate = grid[index][0],
            momentum = grid[index][1],
            number_of_hidden_layers = grid[index][2],
            number_of_nodes_per_layer = grid[index][3]
        )
        start = time.time()   
        
        trainingData = neural_network.fit(X_train_flattened, y_train, epochs=4)
        trainingResults = neural_network.evaluate(X_test_flattened,y_test)

        ellapsedTime = time.time() - start
        
        results['Learning_rate'].append(grid[index][0])
        results['Momentum'].append(grid[index][1])
        results['Number_of_hidden_layers'].append(grid[index][2])
        results['Number_of_nodes_per_layer'].append(grid[index][3])
        results['Time'].append(ellapsedTime)
        results['training_accuracies'].append(trainingData.history['accuracy'])
        results['training_losses'].append(trainingData.history['loss'])
        results['test_loss'].append(trainingResults[0])
        results['test_acc'].append(trainingResults[1])
        
        # update the current max accuracy
        if len(results['max_acc']) == 0:
            results['max_acc'].append(trainingResults[1])
        elif (trainingResults[1] > results['max_acc'][-1]):
            results['max_acc'].append(trainingResults[1])
        else:
            results['max_acc'].append(results['max_acc'][-1])
        
    df_results = pd.DataFrame(results)
    return df_results




def bayesian_optimization(data, search_space, img_path, acquisition_func="ei",iterations=20, epochs=4,continuous=True, fixed_num_neurons=False, random=False):
    """Run the bayesian optimization with the given parameters on x iteration  

    Args:
        data (array): The dataSet
        search_space (dict): The search space containing the hyperparameters
        img_path (string): File path in which to store heat map images
        acquisition_func (str, optional): The type of acquisition function used. Defaults to "ei".
        iterations (int, optional): Number of iterations to perform. Defaults to 20.
        epochs (int, optional): Number of epochs to perform. Defaults to 4.
        continuous (bool, optional): Parameter to indicate if we are using a continuous or a grid search space. Defaults to True.
        fixed_num_neurons (bool, optional): Boolean to indicate if the number of hidden neurons is fixed or not. Defaults to False.
        random (bool, optional): Parameter to indicate that if you want to perform a bayesian opt or a continuous random opt. Defaults to False.

    Returns:
        array : Array containing the dataframes of the runs and hitmaps results
    """

    stored_results = {
        'Learning_rate' : [],
        'Momentum' : [],
        'Number_of_hidden_layers' : [],
        'Number_of_nodes_per_layer' : [],
        'Time' : [],
        'training_accuracies' : [],
        'training_losses' : [],
        'test_loss' : [],
        'test_acc' : [],
        'max_acc' : []
    }
    
    if continuous==True:
        generate_params = get_random_params_continuous
    else:
        generate_params = get_random_params_grid
    
    x = []
    y = []
    
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-10, n_restarts_optimizer=10,normalize_y=True)


    for i in range(iterations):
        print("Iteration Number:",i)
        
        start = time.time()

        
        if random==True or len(x) < 2:
            new_X = generate_params(search_space, fixed_neurons_per_layer=fixed_num_neurons)
        else:
            new_X = aquisition(gpr, y, generate_params, search_space, acquisition_func, fixed_num_neurons)        

        training_data, training_results = bayesian_evalutation(data,new_X,epochs)
        new_y = training_results[1]

        x.append(new_X)
        y.append(new_y)

        gpr.fit(x,y)

        ellapse_time = time.time()-start;

        params = get_params_evaluation(new_X)
        stored_results['Learning_rate'].append(params[0])
        stored_results['Momentum'].append(params[1])
        stored_results['Number_of_hidden_layers'].append(params[2])
        stored_results['Number_of_nodes_per_layer'].append(params[3])
        stored_results['Time'].append(ellapse_time)
        stored_results['training_accuracies'].append(training_data.history['accuracy'])
        stored_results['training_losses'].append(training_data.history['loss'])
        stored_results['test_loss'].append(training_results[0])
        stored_results['test_acc'].append(training_results[1])
        stored_results['max_acc'].append(max(y))

        if i%2==1:
            heat_map_data(gpr, x, y, i, img_path)
    
    df_results = pd.DataFrame(stored_results)

    heat_map_df = heat_map_data(gpr, x, y, i, img_path)

    return df_results, heat_map_df



def bayesian_evalutation(data, params, epochs):
    """Evaluate the neural network with a given set of parameters

    Args:
        data (array): The dataset
        params (array): The set of hyperparameters  
        epochs (int): The number of epochs to perform

    Returns:
        array: Array containing the results dataframe
    """
    X_train_flattened = data[0]
    y_train = data[1]
    X_test_flattened = data[2]
    y_test = data[3]

    params_eval = get_params_evaluation(params)

    learning_rate = params_eval[0]
    momentum = params_eval[1]
    number_of_hidden_layers = params_eval[2]
    number_of_nodes_per_layer = params_eval[3]
    
    neural_network = create_neural(
        learning_rate = learning_rate,
        momentum= momentum,
        number_of_hidden_layers= number_of_hidden_layers,
        number_of_nodes_per_layer= number_of_nodes_per_layer
    )

    print("Hyperparameters:", learning_rate, momentum, number_of_hidden_layers, number_of_nodes_per_layer)
    
    training_data = neural_network.fit(X_train_flattened, y_train, epochs=epochs)
    results = neural_network.evaluate(X_test_flattened,y_test)
    
    return training_data, results    


def aquisition(gpr, y, generate_params, search_space, acquisition_func, fixed_num_neurons):
    """Function that determines the next set of hyperparameters to test

    Args:
        gpr (Gaussian Process Regressor): The Gaussian process regressor model from sklearn module
        y (array): List of accuracies of the testes points
        generate_params (func): Function that generates a new hyperparameter set
        search_space (dict): The searchspace containing the hyperparameters boundaries
        acquisition_func (str): The type of the acquisition function used
        fixed_num_neurons (bool): Indicates if the number of neurons in a layer is changing or not

    Returns:
        array : Array containing the next hyperparameter set 
    """
    X_rand = []
    for i in range(100):
        X_rand.append(generate_params(search_space, fixed_neurons_per_layer=fixed_num_neurons))
    
    y_max = max(y)

    mean, std = gpr.predict(X_rand, return_std = True)

    gamma = [(mean[i]-y_max)/(std[i]+1E-9) for i in range(len(mean))]

    if acquisition_func == "pi":
        acq_vals = [(norm.cdf(gamma[i])) for i in range(len(mean))]
    elif acquisition_func == "ucb":
        kappa = 1.6
        acq_vals = [(mean[i]+kappa*std[i]) for i in range(len(mean))]
    else:
        acq_vals=[std[i]*(gamma[i]*(norm.cdf(gamma[i]))+norm.pdf(gamma[i])) for i in range(len(mean))]

    max_index = np.argmax(acq_vals)

    return X_rand[max_index]


def get_random_params_grid(search_space, fixed_neurons_per_layer=False):
    """Function used to generate random set of hyperparameter from a grid

    Args:
        search_space (dict): The search space containing the hyperparameters boundaries
        fixed_neurons_per_layer (bool, optional): Indicates if the number of neurons per layer is variable or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    hyperparam_combinations = generate_hyperparameters_grid()
    index = randint(0, len(hyperparam_combinations)-1)
    
    X = []
    X.append(math.log10(hyperparam_combinations[index][0]))
    X.append(hyperparam_combinations[index][1])
    X.append(hyperparam_combinations[index][2])
    for i in range(hyperparam_combinations[index][2]):
        X.append(int(math.log2(hyperparam_combinations[index][3][i])))
    return X

def get_random_params_continuous(search_space, fixed_neurons_per_layer=False):
    learning_rate=uniform(search_space['learning_rate'][0],search_space['learning_rate'][1])
    momentum=uniform(search_space['momentum'][0],search_space['momentum'][1])
    num_hidden_layers=randint(search_space['num_hidden_layers'][0],search_space['num_hidden_layers'][1])
    if fixed_neurons_per_layer:
        num_neurons_per_layer=[5]
    else:
        num_neurons_per_layer = []
        max_num_hidden_layers = search_space['num_hidden_layers'][1]
        for i in range(max_num_hidden_layers):
            if i < num_hidden_layers:
                num_neurons_per_layer.append(randint(search_space['num_neurons_per_layer'][0],search_space['num_neurons_per_layer'][1]))
            else:
                num_neurons_per_layer.append(0)

    X = []
    X.append(learning_rate)
    X.append(momentum)
    X.append(num_hidden_layers)
    for i in range(len(num_neurons_per_layer)):
        X.append(num_neurons_per_layer[i])
        
    return X

def get_params_evaluation(params_bayesian):
    """Transform the set of hyperparameter from the bayesian format to the neural network format

    Args:
        params_bayesian (array): Array containing the bayesian parameters   

    Returns:
        array: Array of hyperparameters
    """
    learning_rate = 10**params_bayesian[0]
    momentum = params_bayesian[1]
    num_hidden_layers = params_bayesian[2]
    num_nodes_per_layer = [2**params_bayesian[3+i] for i in range(num_hidden_layers)]
    num_nodes_per_layer.append(10)

    params = [learning_rate, momentum, num_hidden_layers, num_nodes_per_layer]
    return params

def heat_map_data(gpr, X, y, iteration_num, img_path):
    """Calculates the heat maps data

    Args:
        gpr (Gaussian Process Regressor): The gaussian process regressor model from sklearn
        X (array): Hyperparameter sets
        y (array): The accuracy list corresponding to the hyperparameters 
        iteration_num (int): The number of iterations performed
        img_path (str): The saving path of the heat maps

    Returns:
        dataframe: Dataframe containing the heatmap results data
    """
    heat_map = {
        "learning_rate" : [],
        "momentum" : [],
        "predicted_accuracy" : []
    }
    
    max_index = np.argmax(y)
    X_max = X[max_index]
    num_hidden_layers = X_max[2]
    num_neurons_per_layer = [X_max[3+i] for i in range(len(X_max)-3)]
    print(num_neurons_per_layer)

    learning_rate_space = np.linspace(-4,0,50)
    momentum_space = np.linspace(0,1,50)

    learning_rates,momentums = np.meshgrid(learning_rate_space,momentum_space)
    accuracies = []

    for i in range(learning_rates.shape[0]):
        accuracies.append([])
        for j in range(learning_rates.shape[1]):
            param_sample = []
            param_sample.append(learning_rates[i][j])
            param_sample.append(momentums[i][j])
            param_sample.append(num_hidden_layers)
            for k in range(len(X_max)-3):
                param_sample.append(num_neurons_per_layer[k])
            accuracy = gpr.predict([param_sample])[0]
            accuracies[i].append(accuracy)
    

    plot_heat_map(learning_rates, momentums, accuracies, img_path, iteration_num)

    for lr,momentum, acc in zip(learning_rates,momentums,accuracies):
        for i in range(len(lr)):
            heat_map["learning_rate"].append(lr[i])
            heat_map["momentum"].append(momentum[i])
            heat_map["predicted_accuracy"].append(acc[i])
    
    heat_map_df = pd.DataFrame(heat_map)
    return heat_map_df

def plot_heat_map(learning_rates, momentums, accuracies, img_path, iteration_num):
    """Plots the heatmap based on the data

    Args:
        learning_rates (array): Array containing the list of learning rates
        momentums (array): Array containing the list of momentum
        accuracies (array): Array containing the list of accuracies
        img_path (str): Saving path for the heat maps images
        iteration_num (int): The number of iterations performed
    """
    plt.contourf(learning_rates,momentums,accuracies,20,cmap='turbo')
    plt.colorbar()
    plt.xlabel("log( learning rate )")
    plt.ylabel("momentum")
    plt.title(f'Bayesian Accuracy Prediction Iteration {iteration_num+1}')
    plt.savefig(f'{img_path}/heat_map_{iteration_num}.png')
    plt.clf()