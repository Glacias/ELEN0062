import numpy as np
import sys
import itertools
import matplotlib
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def make_data(n_samples, q=0, random_state=None):
    """Generate a dataset of N samples (x_r, x_j_1, ..., x_j_q, y)
    Where :
    - x_r ~ U(-10,10)
    - x_j ~ U(-10,10)
    - y = f(x_r) + (1/10)*eps
        where f(x) = = sin(x_r) ∗ e^(-(x_r)^2/16) and eps ~ N (0, 1)

    Parameters
    ----------
    n_samples : int >0
        The number of samples to generate
    q : int >= 0, optional (default=0)
        The number of irrelevant variables
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, q+1]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    # Build the random generator
    drawer = check_random_state(random_state)
    # Draw the x_r
    # We round up the continious variables to allow computation for specific x
    x_r = np.around(drawer.uniform(-10, 10, n_samples), decimals=1)
    # Draw the eps
    eps = drawer.normal(0, 1, n_samples)
    # Compute the y
    y = np.sin(x_r) * np.exp((-(x_r)**2)/16) + (1/10) * eps

    # Add x_r and the irrelevant variables to X
    X = np.empty((q+1, n_samples))
    X[0] = x_r
    for k in range(1,q+1):
        # We round up the continious variables to allow computation for specific x
        X[k] = np.around(drawer.uniform(-10, 10, n_samples), decimals=1)

    return X.T, y

def protocol1(X, y, regression_model, model_complexity, n_LS):
    """
    Protocol1

    Parameters
    ----------
    X : array of shape [n_samples, q+1]
        The input samples.
    y : array of shape [n_samples]
        The output values.
    regression_model : char
        Can be either "linear" or "nonlinear"
    model_complexity : int > 0 (useless in linear mode)
        Corresponds to the number of neighbors for the kNN
    n_LS : int > 0
        Corresponds to the number of learning samples to divide into

    Return
    ------
    res_err : array of float [n_states]
        Array of residual Error for each x_0 (states)

    bias : array of float [n_states]
        Array of the squared bias for each x_0 (states)

    var_y_pre : array of float [n_states]
        Array of the estimate variance for each x_0 (states)

    x_0_states : array of float [n_states]
        Array of the different states
    """
    # First we look at the different possible values of x_0
    n_samples, n_features = X.shape
    unique = [np.unique(X[:,k]) for k in range(0, n_features)]

    # We compute the possible different state by doing a permutation of the
    # different values taken by the variables
    x_0_states = list(itertools.product(*unique))

    # Number of different states
    n_states = len(x_0_states)

    # Create an array of list
    new_y = np.empty((n_states,),dtype=object)
    for i,v in enumerate(new_y): new_y[i]=[0]

    # Store all different y in each corresponding states
    for index_of_sample in range(0, n_samples) :
        index_of_state = np.where((x_0_states == X[index_of_sample]).all(axis=1))
        new_y[index_of_state[0][0]].append(y[index_of_sample])

    # We compute the residual error (variance) of every x_0 (and the mean for bias)
    res_err = np.zeros(n_states)
    mean_y = np.zeros(n_states)
    for state in range(0, n_states):
            # If there is a least a point y in the state
            if(len(new_y[state])!=1):
                new_y[state].pop(0)
                res_err[state] = np.var(new_y[state])
                mean_y[state] = np.mean(new_y[state])

    # Then we divide into the learning set into multiples samller sets
    splited_X = np.split(X, n_LS)
    splited_y = np.split(y, n_LS)
    mean_y_pre = np.zeros(n_states)
    var_y_pre = np.zeros(n_states)

    # Linear regression
    if(regression_model=="linear"):
        # We train the models
        models = [LinearRegression().fit(splited_X[ls], splited_y[ls]) for ls in range(0, n_LS)]

        # We compute the variance of the predictions made by the models for every x_0
        for j in range(0, n_states):
            state = [np.array(x_0_states[j])]
            # Set of predictions for a specific x_0
            y_pre = [m.predict(state) for m in models]
            mean_y_pre[j] = np.mean(y_pre)
            var_y_pre[j] = np.var(y_pre)

    # Non linear regression
    if(regression_model=="nonlinear"):
        # We train the models
        models = [KNeighborsRegressor(n_neighbors=model_complexity).fit(splited_X[ls], splited_y[ls]) for ls in range(0, n_LS)]

        # We compute the variance of the predictions made by the models for every x_0
        for j in range(0, n_states):
            state = [np.array(x_0_states[j])]
            # Set of predictions for a specific x_0
            y_pre = [m.predict(state) for m in models]
            mean_y_pre[j] = np.mean(y_pre)
            var_y_pre[j] = np.var(y_pre)

    # We compute the bias for every x_0 guessed by the models
    bias = (mean_y - mean_y_pre)**2

    return res_err, bias, var_y_pre, x_0_states

def protocol2(X, y, regression_model, model_complexity, n_LS):
    """
    Parameters
    ----------
    X : array of shape [n_samples, q+1]
        The input samples.
    y : array of shape [n_samples]
        The output values.
    regression_model : char
        Can be either "linear" or "nonlinear"
    model_complexity : int > 0 (useless in linear mode)
        Corresponds to the number of neighbors for the kNN
    n_LS : int > 0
        Corresponds to the number of learning samples to divide into

    Return
    ------
    mean_res_err : float
        Mean of residual Error

    mean_bias : float
        Mean of the squared bias

    mean_var_y_pre : float
        Mean of estimate variance
    """
    # Protocol1
    res_err, bias, var_y_pre, x_0_states = protocol1(X, y, regression_model, model_complexity, n_LS)

    # Average for all x_0 (states)
    return np.mean(res_err), np.mean(bias), np.mean(var_y_pre)

if __name__ == '__main__':
    random_state = 1234

    ##### Q3d (with q=0)
    n_samples = 100000
    q = 0

    ### Linear Regression
    model_complexity = -1 # Not used in linear mode
    n_LS = 100

    X,y = make_data(n_samples, q, random_state)
    res_err, bias, var_y_pre, x_0_states = protocol1(X, y, "linear", model_complexity, n_LS)

    # Residual error
    plt.bar(np.array(x_0_states).T[0], res_err, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the residual error')
    plt.savefig("q3d_res_err_lin.pdf")
    plt.clf()

    # Bias squared
    plt.bar(np.array(x_0_states).T[0], bias, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the bias squared')
    plt.savefig("q3d_bias_lin.pdf")
    plt.clf()

    # Variance
    plt.bar(np.array(x_0_states).T[0], var_y_pre, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the variance')
    plt.savefig("q3d_var_lin.pdf")
    plt.clf()

    # Expected error
    exp_err = res_err + bias + var_y_pre
    plt.bar(np.array(x_0_states).T[0], exp_err, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the expected error')
    plt.savefig("q3d_exp_err_lin.pdf")
    plt.clf()

    ### Non Linear Regression
    model_complexity = 50
    n_LS = 100

    res_err, bias, var_y_pre, x_0_states = protocol1(X, y, "nonlinear", model_complexity, n_LS)

    # Residual error
    plt.bar(np.array(x_0_states).T[0], res_err, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the residual error')
    plt.savefig("q3d_res_err_nonlin.pdf")
    plt.close()

    # Bias squared
    plt.bar(np.array(x_0_states).T[0], bias, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the bias squared')
    plt.savefig("q3d_bias_nonlin.pdf")
    plt.clf()

    # Variance
    plt.bar(np.array(x_0_states).T[0], var_y_pre, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the variance')
    plt.savefig("q3d_bias_nonlin.pdf")
    plt.clf()

    # Expected error
    exp_err = res_err + bias + var_y_pre
    plt.bar(np.array(x_0_states).T[0], exp_err, width=1, linewidth = 0)
    plt.xlabel('x')
    plt.ylabel('Estimation of the expected error')
    plt.savefig("q3d_exp_err_nonlin.pdf")
    plt.clf()

    ##### Q3e

    ### Linear Regression (with q = 0)
    q = 0
    model_complexity = -1 # Not used in linear mode
    n_LS = 100

    range_of_samples = [100, 500, 1000, 5000, 10000, 50000, 100000]

    mean_res_err = np.zeros(len(range_of_samples))
    mean_bias = np.zeros(len(range_of_samples))
    mean_var_y_pre = np.zeros(len(range_of_samples))

    for k in range(0, len(range_of_samples)):
        X,y = make_data(range_of_samples[k], q, random_state)
        mean_res_err[k], mean_bias[k], mean_var_y_pre[k] = protocol2(X, y, "linear", model_complexity, n_LS)

    # Mean residual error
    plt.plot([100, 500, 1000, 5000, 10000, 50000, 100000], mean_res_err)

    # Mean squared bias
    plt.plot([100, 500, 1000, 5000, 10000, 50000, 100000], mean_bias)

    # Mean variance
    plt.plot([100, 500, 1000, 5000, 10000, 50000, 100000], mean_var_y_pre)

    # Expected error
    mean_exp_err = mean_res_err + mean_bias + mean_var_y_pre
    plt.semilogx([100, 500, 1000, 5000, 10000, 50000, 100000], mean_exp_err)
    plt.xlabel('Size of the learning set')
    plt.ylabel('Generalized error for linear regression')
    plt.legend(('Residual error', 'Bias squared', 'Estimation variance', 'Expected error'))
    plt.savefig("q3e_gen_err_lin_size_ls.pdf")
    plt.clf()

    ### Non Linear Regression (with q = 0)
    q = 0
    model_complexity = 5
    n_LS = 50

    range_of_samples = [1000, 5000, 10000, 50000, 100000]

    mean_res_err = np.zeros(len(range_of_samples))
    mean_bias = np.zeros(len(range_of_samples))
    mean_var_y_pre = np.zeros(len(range_of_samples))

    for k in range(0, len(range_of_samples)):
        X,y = make_data(range_of_samples[k], q, random_state)
        mean_res_err[k], mean_bias[k], mean_var_y_pre[k] = protocol2(X, y, "nonlinear", model_complexity, n_LS)

    # Mean residual error
    plt.plot(range_of_samples, mean_res_err)

    # Mean squared bias
    plt.plot(range_of_samples, mean_bias)

    # Mean variance
    plt.plot(range_of_samples, mean_var_y_pre)

    # Expected error
    mean_exp_err = mean_res_err + mean_bias + mean_var_y_pre
    plt.semilogx(range_of_samples, mean_exp_err)
    plt.xlabel('Size of the learning set')
    plt.ylabel('Generalized error for non linear regression')
    plt.legend(('Residual error', 'Bias squared', 'Estimation variance', 'Expected error'))
    plt.savefig("q3e_gen_err_nonlin_size_ls.pdf")
    plt.clf()

    ### Model complexity

    q = 0
    n_samples = 100000
    n_LS = 100

    complexity_samples = [1, 5, 10, 50, 100, 200, 300, 500, 700, 1000]

    mean_res_err = np.zeros(len(complexity_samples))
    mean_bias = np.zeros(len(complexity_samples))
    mean_var_y_pre = np.zeros(len(complexity_samples))

    for k in range(0, len(complexity_samples)):
        X,y = make_data(n_samples, q, random_state)
        mean_res_err[k], mean_bias[k], mean_var_y_pre[k] = protocol2(X, y, "nonlinear", complexity_samples[k], n_LS)

    # Mean residual error
    plt.plot(complexity_samples, mean_res_err)

    # Mean squared bias
    plt.plot(complexity_samples, mean_bias)

    # Mean variance
    plt.plot(complexity_samples, mean_var_y_pre)

    # Expected error
    mean_exp_err = mean_res_err + mean_bias + mean_var_y_pre
    plt.plot(complexity_samples, mean_exp_err)
    plt.xlabel('Model complexity (number of neighbors)')
    plt.ylabel('Generalized error for non linear regression')
    plt.legend(('Residual error', 'Bias squared', 'Estimation variance', 'Expected error'))
    plt.savefig("q3e_gen_err_nonlin_compl.pdf")
    plt.clf()

    ### Number of irrelevant variables added to the problem (linear)

    n_samples = 100000
    model_complexity = -1 #not used in linear
    n_LS = 100

    q_samples = [0, 1]

    mean_res_err = np.zeros(len(q_samples))
    mean_bias = np.zeros(len(q_samples))
    mean_var_y_pre = np.zeros(len(q_samples))

    for k in range(0, len(q_samples)):
        X,y = make_data(n_samples, q_samples[k], random_state)
        mean_res_err[k], mean_bias[k], mean_var_y_pre[k] = protocol2(X, y, "linear", model_complexity, n_LS)

    # Mean residual error
    plt.plot(q_samples, mean_res_err)

    # Mean squared bias
    plt.plot(q_samples, mean_bias)

    # Mean variance
    plt.plot(q_samples, mean_var_y_pre)

    # Expected error
    mean_exp_err = mean_res_err + mean_bias + mean_var_y_pre
    plt.plot(q_samples, mean_exp_err)
    plt.xlabel('Number of irrelevant variables')
    plt.ylabel('Generalized error for non linear regression')
    plt.legend(('Residual error', 'Bias squared', 'Estimation variance', 'Expected error'))
    plt.savefig("q3e_gen_err_lin_q.pdf")
    plt.clf()

    ### Number of irrelevant variables added to the problem (nonlinear)

    n_samples = 100000
    model_complexity = 20
    n_LS = 100

    q_samples = [0, 1]

    mean_res_err = np.zeros(len(q_samples))
    mean_bias = np.zeros(len(q_samples))
    mean_var_y_pre = np.zeros(len(q_samples))

    for k in range(0, len(q_samples)):
        X,y = make_data(n_samples, q_samples[k], random_state)
        mean_res_err[k], mean_bias[k], mean_var_y_pre[k] = protocol2(X, y, "nonlinear", model_complexity, n_LS)

    # Mean residual error
    plt.plot(q_samples, mean_res_err)

    # Mean squared bias
    plt.plot(q_samples, mean_bias)

    # Mean variance
    plt.plot(q_samples, mean_var_y_pre)

    # Expected error
    mean_exp_err = mean_res_err + mean_bias + mean_var_y_pre
    plt.plot(q_samples, mean_exp_err)
    plt.xlabel('Number of irrelevant variables')
    plt.ylabel('Generalized error for non linear regression')
    plt.legend(('Residual error', 'Bias squared', 'Estimation variance', 'Expected error'))
    plt.savefig("q3e_gen_err_nonlin_q.pdf")
    plt.clf()
