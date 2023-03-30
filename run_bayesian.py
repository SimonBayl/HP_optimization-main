from neural import * 
from optimization import *
import os

def run_bayesian(data, search_space,description,  num_trials=5, iterations_per_trial=20,epochs=4,acquisition_func="ei", starting_points=None,continuous=True, random=False, fixed_num_neurons=False):
    """Function used to launch a bayesian optimization


    Args:
        data (array): Array containing the training and testing data
        search_space (dict): Dictionnary containing the hyperparameter boundaries
        description (str): Name of the folder containing the results
        num_trials (int, optional): Number of trials to perform. Defaults to 5.
        iterations_per_trial (int, optional): Number of iterations at each trial. Defaults to 20.
        epochs (int, optional): The number of epochs to perform. Defaults to 4.
        acquisition_func (str, optional): The name of the wanted acquisition function. Defaults to "ei".
        starting_points (array, optional): Array containing the starting hyperparameter set. Defaults to None.
        continuous (bool, optional): Indicates if we want to use a continuous or a grid search space. Defaults to True.
        random (bool, optional): Indicates if we are performing a bayesian optimization or a continuous random grid search. Defaults to False.
        fixed_num_neurons (bool, optional): Indicates if the number of neurons per layer is fixed or not. Defaults to False.
    """
    file_path = f'results/bayesian/{description}'
    try:
        os.mkdir(file_path, 777)
    except:
        print(f'{file_path} already exists')

    context = open(f'{file_path}/context.txt', 'w')
    context.write(f'{description} \n')
    context.write(f'Acquisition: {acquisition_func} \n')
    context.write(f'Continuous: {continuous} \n')
    context.write(f'Random: {random} \n')
    context.write(f'Fixed: {fixed_num_neurons} \n')
    context.write("Parameters:  \n")
    for parameter in search_space:
        context.writelines(f'\t {parameter}: {search_space[parameter]} \n')
    context.close

    try:
        os.mkdir(f'{file_path}/heat_maps_img', 777)
    except:
        print(f'{file_path}/heat_maps_img already exists')
    try:
        os.mkdir(f'{file_path}/heat_maps_data', 777)
    except:
        print(f'{file_path}/heat_maps_data exists')
    try:
        os.mkdir(f'{file_path}/results', 777)
    except:
        print(f'{file_path}/results already exists')

    aggregate_max_acc = {}    

    if starting_points == None:
        print("entered")
        starting_points = []
        for i in range(num_trials):
            starting_points.append(None)

    for i in range(num_trials):
        print(f'Trail Numer: {i}')

        img_file_path = f'{file_path}/heat_maps_img/Trial{i}'
        try:
            os.mkdir(img_file_path, 777)
        except:
            print(f'{img_file_path} already exists')
        
        training_results, heat_map_results = bayesian_optimization(
            data, 
            search_space, 
            img_file_path,   
            acquisition_func=acquisition_func,
            starting_point=starting_points[i],
            iterations=iterations_per_trial,
            epochs=epochs,
            continuous=continuous,
            random=random,
            fixed_num_neurons=fixed_num_neurons
        )

        aggregate_max_acc[f'Trial_{i}'] = training_results["max_acc"]

        training_results.to_csv(f'{file_path}/results/results_trial_{i}.csv', sep=";")
        heat_map_results.to_csv(f'{file_path}/heat_maps_data/heat_map_data_trial_{i}.csv', sep=";")
    
    aggregate_max_acc_df = pd.DataFrame(aggregate_max_acc)
    aggregate_max_acc_df.to_csv(f'{file_path}/aggregate_max_acc.csv', sep=";")