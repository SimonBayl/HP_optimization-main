from tensorflow import keras

def loadDataset(dataset, number_of_pixels_value = 255):
    """Load the images from the dataset

    Args:
        number_of_pixels_value (int): The number of possible values for each pixel
        picture_width (int): Width of the picture
        picture_height (int): Height of the picture
        dataset (tuple(tuple(int))): The dataset from keras

    Returns:
        tuple : Tuples containing the training and testing data
    """
    
    (X_train, y_train), (X_test, y_test) = dataset  #loads image from MNIST dataset
    X_train = X_train / number_of_pixels_value
    X_test = X_test / number_of_pixels_value

    picture_width = X_train.shape[1]
    picture_height = X_train.shape[2]
    
    X_train_flattened = X_train.reshape(len(X_train), picture_width * picture_height)
    X_test_flattened = X_test.reshape(len(X_test), picture_width * picture_height)
    
    return (X_train_flattened, X_test_flattened), (y_train, y_test)



def create_neural(learning_rate = 0.01, momentum = 0, number_of_hidden_layers = 0, number_of_nodes_per_layer = [10]):
    """Function to create a neural network

    Args:
        learning_rate (float, optional): Learning rate of the gradient descent optimizer. Defaults to 0.01.
        momentum (float, optional): Momentum of the gradient descent optimizer. Defaults to 0.
        number_of_hidden_layers (int, optional): Number of layers before the last layer. Defaults to 0.
        number_of_nodes_per_layer (list, optional): List containing the number of nodes per layer (last one included). Defaults to [10].

    Returns:
        keras.Sequential : The parameterized neural network
    """
    
    layers = [keras.layers.Dense(number_of_nodes_per_layer[i], activation='relu') for i in range(number_of_hidden_layers)]
    layers.append(keras.layers.Dense(number_of_nodes_per_layer[-1], activation='softmax'))
    
    neural_network = keras.Sequential(layers)
    
    opt = keras.optimizers.SGD(
        learning_rate = learning_rate,
        momentum= momentum,
    )

    neural_network.compile(
        optimizer= opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return neural_network


