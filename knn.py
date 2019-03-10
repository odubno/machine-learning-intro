import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.utils import shuffle
from collections import defaultdict


def get_columns():
    col_file = open('README', 'r').read()
    return [i.split()[1] for i in col_file.split('\r\n') if len(i.split()) == 2]


def get_df():
    X = pd.read_csv('X.csv', names=range(1, 54))

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


def split_data(df_split, fold=None):
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
    return X_train, y_train, X_test, y_test


def get_l1(row_1, row_2):
    # absolute distance
    return sum(abs(row_1-row_2))


def get_l2(row_1, row_2):
    # euclidean distance
    return np.sqrt(sum((row_1-row_2)**2))


def get_distance(X_train, X_test_row):
    total_rows = len(X_train)
    for row_index in xrange(total_rows):
        absolute_distance = get_l1(X_train[row_index], X_test_row)
        yield (row_index, absolute_distance)


def get_neighbors(X_train, X_test_row, k):
    """
    * calculate the absolute distance i.e. L1 between a single test row and the training set.
    * get the top k number of closest points from X_train that are closest to the X_test_row
    """
    distances = [i for i in get_distance(X_train, X_test_row)]
    sorted_distances = pd.DataFrame(distances).sort_values(1)
    nearest_neighbors = list(sorted_distances[:k][0])
    return nearest_neighbors


def predict(k_points, y_train):
    y_neighbors = [y_train[i] for i in k_points]
    closest_neighbor = max(set(y_neighbors), key=y_neighbors.count)
    return closest_neighbor


def get_knn(X_train, X_test, y_train, k_range):
    k_map = defaultdict(list)
    total_rows = len(X_test)
    for row_index in xrange(total_rows):
        print '%sth iteration out of %s' % (row_index, total_rows)
        X_test_row = X_test[row_index]
        k_neighbors = get_neighbors(X_train, X_test_row, k_range)
        for k in range(1, k_range+1):
            predicted = predict(k_neighbors[:k], y_train)
            k_map[k].append(predicted)
    return k_map


def prediction_accuracy(y_pred, y_test):
    total_predictions = len(y_pred)
    correct_predictions = [i for i in range(total_predictions) if y_pred[i] == y_test[i]]
    return len(correct_predictions) / float(total_predictions)


def knn(df):
    X_train, y_train, X_test, y_test = split_data(df)
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    k_range = 21
    y_pred_map = get_knn(X_train, X_test, y_train, k_range)
    accuracies = []
    for k in range(1, k_range+1):
        accuracy = prediction_accuracy(y_pred_map[k], y_test)
        accuracies.append(accuracy)
    return accuracies


def get_plot(accuracies):
    length = range(1, len(accuracies) + 1)
    plt.plot(length, accuracies)
    plt.xticks(length)
    plt.title('Prediction Accuracy (function of k)')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    df = get_df()
    accuracies = knn(df)
    print accuracies
    get_plot(accuracies)
