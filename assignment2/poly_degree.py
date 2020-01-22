import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

MIN_DEGREE = 0
MAX_DEGREE = 10


def polyfit(x, y):
    """
    Uses NumPy's polyfit function to determine the coefficients of the polynomial
    for each degree using the least-squares estimate. The R^2 value is computed
    for each degree in an attempt to determine which degree fits best. 

    ARGS
        x:  input x data
        y:  input y data
    RETURNS 
        best_fit:   a dictionary containing the degree and R^2 value of the degree 
                    with the best fit to the input data
    """
    # Find the value of k between 0 and 10
    best_fit = {"k": None, "r_squared": -np.inf, "coefs": None}
    r_squares = []
    for i in range(MIN_DEGREE, MAX_DEGREE + 1):
        # Use polyfit to determine the coefficients for this order
        output = np.polyfit(x, y, i, full=True)

        coefs = output[0]
        residuals = output[1]
        rank = output[2]
        singular_values = output[3]
        rcond = output[4]

        # Plot the resulting polynomial on top of the input data
        poly_fitted = np.poly1d(coefs)
        y_new = poly_fitted(x)

        # Find R^2 value
        y_avg = np.average(y)
        ssreg = np.sum((y_new-y_avg)**2)
        sstot = np.sum((y-y_avg)**2)
        r_squared = ssreg / sstot
        r_squares.append(r_squared)

        plt.plot(x, y, 'o', markersize=3, label='input data')
        plt.plot(x, y_new, label="fitted polynomial")
        plt.legend(loc="upper right")
        plt.title("Polyfit Results for k={}".format(i))
        #plt.savefig("polyfit_{}.png".format(i))
        #plt.show()
        plt.clf()
        
    plt.plot(np.linspace(0, 10, 11), r_squares)
    plt.title("$R^2$ Value for Each Value of k (Polyfit)")
    plt.xlabel("Value of k")
    plt.ylabel("$R^2$")
    #plt.savefig("polyfit_r2.png")
    plt.show()


def linear_regression(x, y):
    """
    Uses SciKit-Learn's Stochastic Gradient Descent model to fit a polynomial 
    to the input data. This is a machine learning approach where the data is 
    split into training and testing data. The RMSE is then calculated for each
    order, as well as the resulting test score. These numbers are then plotted
    to provide a visualization of the results of the regression. 

    ARGS
        x:  input x data
        y:  input y data
    RETURNS
        nothing  
    """
    RMSE = []
    test_score = []
    r_squares = []
    test_set_fraction=0.2

    # Reshape the data to have one column and n rows
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_set_fraction)
    
    # Make the pipeline for the model for each degree
    for degree in range(MIN_DEGREE, MAX_DEGREE + 1):
        model = make_pipeline(
            PolynomialFeatures(degree), 
            SGDRegressor(
                fit_intercept=False, 
                learning_rate='constant',
                eta0=0.000001, 
                alpha=0.01, 
                max_iter=5000, 
                average=False, 
                penalty='l1',
                loss='epsilon_insensitive',
            )
        )
        # Fit the model to the training data and predict the output
        model.fit(X_train, y_train)
        y_pred = np.array(model.predict(X_test))

        # Calculate the root mean squared error and get the test score
        RMSE.append(np.sqrt(np.sum(np.square(y_pred-y_test))))
        test_score.append(model.score(X_test, y_test))
        
        # Find R^2 value
        r_squared = r2_score(y_test, y_pred)
        r_squares.append(r_squared)

    # Plot the resulting metrics
    plt.plot(np.linspace(0, 10, num=11), RMSE)
    plt.title("Root mean squared error for SGD")
    plt.xlabel("Value of k")
    #plt.savefig("sgd_rmse.png")
    plt.show()
    plt.clf()
    plt.plot(np.linspace(0, 10, num=11), test_score)
    plt.title("Test score for SGD")
    plt.xlabel("Value of k")
    #plt.savefig("sgd_test_score.png")
    plt.show()
    plt.clf()
    plt.ylim(top=1, bottom=0)
    plt.plot(np.linspace(0, 10, num=11), r_squares)
    plt.title("$R^2$ Value for SGD")
    plt.xlabel("Value of k")
    #plt.savefig("sgd_r2.png")
    plt.show()
    plt.clf()


def gradient_descent(x, y):
    """
    Implement gradient descent with a learning rate. 

    Args:
        x:  input x data
        y:  input y data
    """
    # Determine values for overflow
    float_info = np.finfo(np.longdouble)
    min_float = float_info.min
    max_float = float_info.max

    # Set constants
    learning_rate = 0.001
    max_epochs = 20000
    r_squares = []
    best_fit = {"k": None, "r_squared": -np.inf, "coefs": None}
    y = np.longdouble(y)
    for i in range(MIN_DEGREE, MAX_DEGREE + 1):
        # Decrease the learning rate as the order of the polynomial increases
        learning_rate = learning_rate / 10

        # Build the Vandermonde matrix
        X = np.vander(x, N=i+1)

        # Initialize a to a random matrix
        a = np.random.normal(0, 1, i+1)

        # Cast to 64 bit floats
        X = np.longdouble(X)
        a = np.array(np.longdouble(a)).reshape(-1)
        for epoch in range(1, max_epochs+1):
            # Calculate the error for this epoch
            current_error = np.matmul(X, a) - y

            # Check if any number is larger than the overflow
            current_error_bool = (current_error >= 0.5*max_float) | (current_error <= 0.5*min_float)
            if current_error_bool.any():
                break
            
            # Calculate the gradient for this epoch
            current_gradient = 2*np.matmul(X.transpose(), current_error)
            
            # Check if the gradient has any NaNs or Infs
            current_gradient_bool = (current_gradient == np.nan) | (current_gradient == np.inf) | (current_gradient == -np.inf)
            if current_gradient_bool.any():
                break

            # Update the coefficients
            a -= learning_rate*current_gradient
        
        # Plot the resulting polynomial on top of the input data
        poly_fitted = np.poly1d(a)
        y_new = np.longdouble(poly_fitted(x))

        # Find R^2 value to determine the best k
        y_avg = np.longdouble(np.average(y))
        ssreg = np.sum((y_new-y_avg)**2)
        sstot = np.sum((y-y_avg)**2)
        r_squared = ssreg / sstot
        r_squares.append(r_squared)

        # Plot the fitted curve
        plt.plot(x, y, 'o', markersize=3, label='input data')
        plt.plot(x, y_new, label="fitted polynomial")
        plt.legend(loc="upper right")
        plt.title("Gradient Descent Results for k={}".format(i))
        #plt.savefig("grad_descent_{}.png".format(i))
        #plt.show()
        plt.clf()

    # Plot the R^2 graph
    plt.ylim(top=1, bottom=0)
    plt.plot(np.linspace(0, 10, 11), r_squares)
    plt.title("$R^2$ Value for Each Value of k (Gradient Descent)")
    plt.xlabel("Value of k")
    plt.ylabel("$R^2$")
    #plt.savefig("gradient_descent_r2.png")
    plt.show()


if __name__=='__main__':
    # read the dataframe df from the csv file
    df = pd.read_csv("polydata.csv")
    # extract x and y values from the corresponding columns in the dataframe 
    x = df.loc[:,'x'].values
    y = df.loc[:,'y'].values
    # now x and y contain the data values from a polynomial

    # Use NumPy's polyfit function to determine k
    polyfit(x, y)

    # Use my Gradient Descent implementation to determine k
    gradient_descent(x, y)

    # Use SciKit-Learn's SGD regression in a pipeline to determine k
    linear_regression(x, y)


    