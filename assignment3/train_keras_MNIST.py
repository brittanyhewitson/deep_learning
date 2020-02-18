import scipy.special
import keras
import os

import numpy as np
import pandas as pd

from timeit import default_timer as timer
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import np_utils

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_nn(**kwargs):
    """
    Run the neural network for the given parameters

    Adapted from the code provided in lecture
    """
    
    # Start the timer
    start_t = timer()

    # Number of input, hidden, and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # Set parameters
    learning_rate = kwargs["learning_rate"]
    optimizer = kwargs["optimizer"]
    batch_size = kwargs["batch_size"]
    epochs = kwargs["epochs"]

    # Create the Keras model
    model = Sequential()
    model.add(Dense(
                hidden_nodes, 
                activation='sigmoid', 
                input_shape=(input_nodes,), 
                bias=False,
            )
        )
    model.add(Dense(
                output_nodes,
                activation='sigmoid',
                bias=False,
            )
        )
    # Print the model summary
    model.summary()

    # Set the optimizer 
    if optimizer == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adagrad":
        opt = optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == "adadelta":
        opt = optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == "adamax":
        opt = optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == "nadam":
        opt = optimizers.Nadam(learning_rate=learning_rate)
    # Default optimizer is adam
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)

    # Define the error criterion, optimizer, and an optional metric to monitor during training
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    # Load the mnist training data CSV
    df = pd.read_csv("mnist_csv/mnist_train.csv", header=None)

    # Columns 1-784 are the input values
    x_train = np.asfarray(df.loc[:, 1:input_nodes].values)
    x_train /= 255.0

    # Column 0 is the desired label
    labels = df.loc[:, 0].values

    # Convert labels to one-hot vectors
    y_train = np_utils.to_categorical(labels, output_nodes)

    # Train the neural network
    # Train the model
    model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1
    )

    # Save the model
    model.save('MNIST_3layer_keras.h5')
    print('model saved')

    # Test the model

    # Load the MNIST test data CSV file into a list 
    test_data_file = open('mnist_csv/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # Scorecard for how well the network performs, initially empty
    scorecard = []

    # Go through all the data in the test data set, one by one
    for record in test_data_list:
        # Split the record by the commas
        data_sample = record.split(',')
        
        # Correct answer is first value
        correct_label = int(data_sample[0])

        # Scale and shift the inputs
        inputs = np.asfarray(data_sample[1:]) / 255.0

        # Make prediction
        outputs = model.predict(np.reshape(inputs, (1, len(inputs))))

        # The index of the highest value corresponds to the label 
        label = np.argmax(outputs)

        # Append correct or incorrect to list
        if label == correct_label:
            # Network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # Netowrk's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        pass

    # Calculate the accuracy
    scorecard_array = np.asarray(scorecard)
    accuracy = scorecard_array.sum()/scorecard_array.size
    print('accuracy = {}'.format(accuracy))


    # Stop the timer
    end_t = timer()
    execution_time = end_t-start_t
    print('elapsed time = {}'.format(execution_time))

    output = {'accuracy': accuracy, 'execution_time': execution_time}
    return output

if __name__=='__main__':
    # Run with the default parameters
    run_nn(
        optimizer="adam",
        learning_rate=0.001,
        batch_size=32,
        epochs=5
    )