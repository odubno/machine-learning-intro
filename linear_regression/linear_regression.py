import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def get_data():
    X_test = pd.read_csv('hw1-data/X_test.csv')
    X_train = pd.read_csv('hw1-data/X_train.csv')
    y_test = pd.read_csv('hw1-data/y_test.csv')
    y_train = pd.read_csv('hw1-data/y_train.csv')
    return X_test, X_train, y_test, y_train


def df_plot(Wrr_array, df_array):
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    labels = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7"]
    for i in range(0, Wrr_array.shape[1]):
        plt.plot(df_array, Wrr_array[:, i], color=colors[i])
        plt.scatter(df_array, Wrr_array[:, i], color=colors[i], label=labels[i], s=1, linewidths=.5)
    plt.xlabel(r"df($\lambda$)")
    lgnd = plt.legend(loc='lower left', fontsize=10)
    # update handle size
    _ = [i.set_sizes([12.0]) for i in lgnd.legendHandles]
    plt.show()


def get_sigma(X_train):
    """ Get square matrix from original matrix
    SVD
        Decompose matrix X into a product of three matrices; two orthogonal matrices U and V and a diagonal matrix s.

                n: number of rows
                k: number of variables

        U:
            (n x k) matrix.
            same dimenstions as your original matrix.
            left-singular vectors of the original matrix.
        s (Sigma):
            (k x k) matrix.
            diagonal matrix is a square and has (k x k) dimension.
            vector of singular values of the original matrix.
        V (V^T):
            (k x k)^T matrix. transpose of (k x k) matrix
            right-singular vectors of the original matrix.
    """
    _, s, _ = np.linalg.svd(X_train)
    return s


def solve_rr(X_train, y_train):
    Wrr_list = []
    df_list = []
    sigma = get_sigma(X_train)
    for lam in range(0, 5000):

        # calculate ridge estimates
        Wrr = get_Wrr(X_train, y_train, lam)
        Wrr_list.append(Wrr)

        # calculate degrees of freedom
        df = sum(sigma**2 / (sigma**2 + lam))
        df_list.append(df)

    return np.array(Wrr_list), np.array(df_list)


def get_Wrr(X_train, y_train, lam):
    """ Get optimum estimate for ridge regression
    :param X_train: (pandas.DataFrame) Matrix
    :param y_train: (pandas.DataFrame) Vector
    :param lam: (int) lambda
    :return: list of estimates per feature
    """
    X_transpose = np.transpose(X_train)
    X_transpose_X = np.dot(X_transpose, X_train)
    lambda_identity = np.identity(len(X_transpose_X)) * lam
    matrix_sum = X_transpose_X + lambda_identity
    matrix_inverse = np.linalg.inv(matrix_sum)
    X_transpose_y = np.dot(X_transpose, y_train)
    Wrr = np.dot(matrix_inverse, X_transpose_y)
    return Wrr


def plot_rmse(rmse_list, p=1, single_plot=False):
    if single_plot:
        plt.figure()
    plt.plot(range(len(rmse_list)), rmse_list)
    plt.scatter(range(len(rmse_list)), rmse_list, s=1, linewidths=.5, label='p=%s' % p)
    plt.title('RMSE vs $\lambda$ on Test Set)')
    plt.xlabel('$\lambda$')
    plt.ylabel('RMSE')
    # update handle size for legend
    lgnd = plt.legend(loc='upper left')
    _ = [i.set_sizes([40.0]) for i in lgnd.legendHandles]
    if single_plot:
        plt.show()


def get_rmse_list(X_test, y_actual, Wrr_array, lam):
    rmse_list = []
    for l in range(0, lam):
        rr_vals = Wrr_array[l]
        y_predicted = np.dot(np.array(X_test), rr_vals)
        rmse_value = mean_squared_error(np.array(y_actual), y_predicted) ** .5
        rmse_list.append(rmse_value)
    return rmse_list


def add_p_order(matrix, p):
    """
    :param matrix: (numpy.array)
    :param p: (int) pth polynomial order
    """
    if p == 1:
        return matrix
    elif p == 2:
        # square the original matrix and append it to the original matrix
        a = matrix
        b = matrix[:, 0:6]**2
        return np.hstack((a, b))
    elif p == 3:
        # square the original matrix and append it to the original matrix
        a = matrix
        b = matrix[:, 0:6]**2
        # cube the original matrix and append it to the previous resulting matrix.
        c = matrix[:, 0:6]**3
        return np.hstack((a, b, c))


def p_order_regression(X_test, X_train, y_test, y_train):
    plt.figure()
    for p in [1, 2, 3]:
        X_train_poly = add_p_order(np.array(X_train), p)
        X_test_poly = add_p_order(np.array(X_test), p)
        Wrr_array, df_array = solve_rr(X_train_poly, y_train)
        rmse_list = get_rmse_list(X_test_poly, y_test, Wrr_array, 100)
        print 'p:%s - optimal lambda: %s, min RMSE: %s' % (p, rmse_list.index(min(rmse_list)), min(rmse_list))
        plot_rmse(rmse_list, p)
    plt.show()


if __name__ == '__main__':
    X_test, X_train, y_test, y_train = get_data()
    Wrr_array, df_array = solve_rr(X_train, y_train)
    df_plot(Wrr_array, df_array)
    rmse_list = get_rmse_list(X_test, y_test, Wrr_array, 50)
    plot_rmse(rmse_list, single_plot=True)
    p_order_regression(X_test, X_train, y_test, y_train)
    print 'done'