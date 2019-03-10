import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.utils import shuffle
from scipy.special import expit
from collections import defaultdict


def get_columns():
    col_file = open('README', 'r').read()
    return [i.split()[1] for i in col_file.split('\r\n') if len(i.split()) == 2]


def get_df(scale):
    X = pd.read_csv('X.csv', names=range(1, 54))

    if scale:
        # scaling all data to be between 0 and 1
        x_values = X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x_values)
        X = pd.DataFrame(x_scaled)

    y = pd.read_csv('y.csv', names=['y'])
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    df = pd.concat([X, y], axis=1)
    df_shuffle = shuffle(df)
    df_shuffle.reset_index(inplace=True, drop=True)
    return df_shuffle


def cross_val_split(df, kfold):
    """ split data into k groups
    :param df: (pandas.DataFrame)
    :param kfolds: (int) number of folds
    :return:
        a list of k dataframes sampled from original df.
    """
    df_copy = df
    fold_size = int(len(df)/kfold)
    df_split = []
    for k in range(kfold):
        df_shuffle = shuffle(df_copy)
        df_sample = df_shuffle[:fold_size]
        # remove sampled rows from df_copy
        df_copy = df_copy[fold_size:]
        df_split.append(df_sample)
    return df_split


def split_data(df_split, fold=None, transform=False):
    # use single fold for testing
    if isinstance(fold, int):
        test = df_split[fold]

        # use the rest of the folds for training ignoring k-th index, because that is used for testing
        train_dfs = [df for index, df in enumerate(df_split) if index != fold]
        train = pd.concat(train_dfs)
    else:
        data_size = int(len(df_split)*.20)
        test = df_split[:data_size]
        train = df_split[data_size:]

    # split data into X and y
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    if transform:
        X_train[54] = 1
        X_test[54] = 1
        y_train.loc[y_train == 0] = -1
        y_test.loc[y_test == 0] = -1
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def get_update(X_train, y_train, sigmoid_list):
    total_rows = X_train.shape[0]
    for row_index in xrange(total_rows):
        yield y_train[row_index] * (1 - sigmoid_list[row_index]) * X_train[row_index]


def get_objective_func(X_train, y_train, w):
    """
    expit is the inverse of the logit function.
    """
    total_rows = X_train.shape[0]
    for row_index in xrange(total_rows):
        x = y_train[row_index] * np.dot(X_train[row_index], w)
        sigmoid = expit(x)
        yield sigmoid


def get_objetive_plot(values):
    plt.title('Objective Function (Steepest Ascent) t = 1,...,1000 for 10 k fold')
    plt.xlabel('# of iterations')
    for i in range(len(values)):
        plt.plot(values[i])
    plt.show()


def get_delta(X_train, y_train, w):
    total_rows, total_cols = X_train.shape
    vector = [0] * total_cols
    for row_index in xrange(total_rows):
        x = y_train[row_index] * np.dot(X_train[row_index], w)
        sigma = expit(x)
        vector = vector + y_train[row_index] * (1 - sigma) * X_train[row_index]
    return vector


def get_square_inverse(X_train, w):
    total_rows, total_cols = X_train.shape
    square_matrix = np.zeros((total_cols, total_cols))
    for row_index in xrange(total_rows):
        sigma = expit(np.dot(X_train[row_index], w))
        outer_product = np.outer(X_train[row_index], X_train[row_index])
        u = sigma * (1 - sigma) * outer_product
        square_matrix = square_matrix + u
    inv_square_matrix = np.linalg.inv(-square_matrix)
    return inv_square_matrix


def get_steepest_ascent():
    df = get_df(scale=True)
    k = 10
    df_split = cross_val_split(df, k)
    likelihood_list = []
    for df_index in range(k):
        X_train, y_train, X_test, y_test = split_data(df_split, df_index, transform=True)
        likelihood_values = [i for i in get_ascent_likelihood(np.array(X_train), np.array(y_train), df_index)]
        likelihood_list.append(likelihood_values)
    return likelihood_list


def get_ascent_likelihood(X_train, y_train, k):
    step_size = 0.01 / 4600
    w = [0] * X_train.shape[1]
    for t in xrange(1, 1001):
        print 'k=%s; %sth iteration out of %s' % (k, t, 1001)
        sigmoid_list = [i for i in get_objective_func(X_train, y_train, w)]
        log_likelihood = sum([np.log(i) for i in sigmoid_list])
        u = sum([i for i in get_update(X_train, y_train, sigmoid_list)])
        w = w + (step_size * u)
        yield log_likelihood


def get_newton_likelihood(X_train, y_train, k):
    w = [0] * X_train.shape[1]
    objective_values = []
    for t in xrange(1, 101):
        print 'k: %s; %sth iteration out of %s' % (k, t, 101)
        sigmoid_list = sum([i for i in get_objective_func(X_train, y_train, w)])
        objective_values.append(sigmoid_list)
        square_inv = get_square_inverse(X_train, w)
        delta = get_delta(X_train, y_train, w)
        np_dot = np.dot(square_inv, delta)
        w -= np_dot
    return objective_values, w


def get_newtown_method():
    df = get_df(scale=True)
    k = 10
    df_split = cross_val_split(df, k)
    objective_results = []
    accuracy_results = defaultdict(list)
    for df_index in range(k):
        X_train, y_train, X_test, y_test = split_data(df_split, df_index, transform=True)
        objective_values, w = get_newton_likelihood(X_train, y_train, df_index)
        accuracy, tp, fp, tn, fn = get_accuracy(w, X_test, y_test)
        accuracy_results['accuracy'].append(accuracy)
        accuracy_results['tp'].append(tp)
        accuracy_results['fp'].append(fp)
        accuracy_results['tn'].append(tn)
        accuracy_results['fn'].append(fn)
        objective_results.append(objective_values)
    accuracy_matrix = {k: sum(v) / len(v) for k, v in accuracy_results.iteritems()}
    return objective_results, accuracy_matrix


def get_y_pred(X_test, w):
    rows_num = len(X_test)
    for row_index in range(rows_num):
        x = np.dot(X_test[row_index], w)
        sigma = expit(x)
        if sigma < 0.5:
            yield -1
        else:
            yield 1


def get_accuracy(w, X_test, y_test):
    y_pred = [i for i in get_y_pred(X_test, w)]
    return get_confusion_matrix(y_test, y_pred)


def get_confusion_matrix(y_pred, y_test):
    df = pd.DataFrame([list(y_pred), list(y_test)]).transpose()
    total_correct = len(df[df[0] == df[1]])
    tp = len(df[(df[0] == 1) & (df[1] == 1)])
    fp = len(df[(df[0] == 1) & (df[1] == -1)])
    tn = len(df[(df[0] == -1) & (df[1] == -1)])
    fn = len(df[(df[0] == -1) & (df[1] == 1)])
    return float(total_correct) / len(y_pred), tp, fp, tn, fn


def get_newton_objective_plot(values):
    plt.title("Objective Function (Newton Method) t = 1,...,100 for 10 k fold")
    plt.xlabel("# of iterations")
    for i in range(len(values)):
        plt.plot(values[i])
    plt.show()


if __name__ == '__main__':
    objective_values = get_steepest_ascent()
    get_objetive_plot(objective_values)

    objective_results, accuracy_matrix = get_newtown_method()
    get_newton_objective_plot(objective_results)
