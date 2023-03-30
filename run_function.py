from neural import * 
from optimization import *
import os

def run_bayesian(
        data, 
        search_space,
        description,  
        num_trials=5, 
        iterations_per_trial=20,
        epochs=4,
        acquisition_func="ei", 
        continuous=True, 
        random=False, 
        fixed_num_neurons=False):

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


    for i in range(num_trials):
        print(f'Trial Numer: {i}')

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