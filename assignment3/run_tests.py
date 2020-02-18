import os 
import sys
import copy
import logging
import pandas as pd
from train_keras_MNIST import run_nn

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    stream=sys.stderr, 
    level=logging.INFO
)

'''
This script was used to run train_keras_MNIST.py for all requested parameters
for all possible optimizers. train_keras_MNIST.py is the script that has all
of the neural network setup, training, and testing in it. 
'''


def main(**kwargs):
    """
    Script used to run the NN code for multiple optimizers and parameters

    """
    # Set the default parameters
    default_params = {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
    }

    # Create the test parameters
    optimizers = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [1, 4, 16, 64]
    epochs = [5, 10, 15, 20]
    variable_params = {
        "learning_rate": learning_rates,
        "batch_size": batch_sizes,
        "epochs": epochs
    }

    # Run the test on each parameter
    for optimizer in optimizers:
        # Create the dataframe
        results = pd.DataFrame()
        default_params["optimizer"] = optimizer
        for test_param, values in variable_params.items():
            for test_constraint in values:
                logging.info("Testing {}={}".format(test_param, test_constraint))
                # Copy the default parameters
                test_params = copy.deepcopy(default_params)

                # Update this field for the test
                test_params[test_param] = test_constraint
                
                # Run the neural network
                output = run_nn(**test_params)

                # Create dataframe and append to results
                test_params["accuracy"] = output["accuracy"]
                test_params["execution_time"] = output["execution_time"]
                tmp_df = pd.DataFrame([test_params])
                results = pd.concat([results, tmp_df], ignore_index=True)

        results.to_csv("results/test_result_{}.csv".format(optimizer), index=False)


if __name__=='__main__':
    main()