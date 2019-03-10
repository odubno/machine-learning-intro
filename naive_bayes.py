import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
import math



def get_columns():
    col_file = open('README', 'r').read()
    return [i.split()[1] for i in col_file.split('\r\n') if len(i.split()) == 2]


def get_df(scale=False, scale_by=None):
    X = pd.read_csv('X.csv', names=range(1, 54))

    if scale:
        # scaling all data to be between 0 and 1
        x_values = X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x_values)
        X = pd.DataFrame(x_scaled)

    if scale_by:
        X = X * scale_by

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


def bernoulli_likelihood(feature_col, y_train):
    sum_1 = sum(y_train)
    sum_0 = len(y_train) - sum_1
    total_1 = sum([i[0] for i in zip(feature_col, y_train) if i[1] == 1])
    total_0 = sum([i[0] for i in zip(feature_col, y_train) if i[1] == 0])
    return [total_0/sum_0, total_1/sum_1]


def get_mle(X_train, y_train):
    mle_list = []
    for col in xrange(0, X_train.shape[1]):
        mle_list.append(bernoulli_likelihood(X_train.iloc[:, col], y_train))
    return mle_list


def get_prior(y_train):
    count_0 = len([i for i in y_train if i == 0])
    count_1 = len([i for i in y_train if i == 1])
    length = float(len(y_train))
    return [count_0/length, count_1/length]


def poisson_pdf(mle_list, feature, y_label):
    # bernoulli
    # dist = mle_list[y_label] ** feature * (1 - mle_list[y_label]) ** (1 - feature)
    p = mle_list[y_label]
    x = int(round(feature))
    dist = ((math.e**-p) * (p**x)) / math.factorial(x)
    return dist


def get_likelihood(priors, x_test_row, mle_list, y_label):
    maximum_likelihood = 1
    for i, feature_val in enumerate(x_test_row):
        likelihood = poisson_pdf(mle_list[i], feature_val, y_label)
        if not likelihood:
            continue
        maximum_likelihood = maximum_likelihood * likelihood
    likelihood = maximum_likelihood * priors[y_label]
    return likelihood


def get_predictions(X_test, priors, mle_list):
    predicted = []
    for i in range(X_test.shape[0]):
        x_test_row = X_test.iloc[i]
        likelihood_0 = get_likelihood(priors, x_test_row, mle_list, 0)
        likelihood_1 = get_likelihood(priors, x_test_row, mle_list, 1)
        likelihoods = [likelihood_0, likelihood_1]
        predicted.append(likelihoods.index(max(likelihoods)))
    return predicted


def nb_classifier(k=10):
    df = get_df(scale=True, scale_by=100)
    df_split = cross_val_split(df, k)
    accuracy_list = []
    accuracy_matrix = defaultdict(list)
    for df_index in range(k):
        X_train, y_train, X_test, y_test = split_data(df_split, df_index)
        mle_list = get_mle(X_train, y_train)
        priors = get_prior(y_train)
        predictions = get_predictions(X_test, priors, mle_list)
        accuracy, tp, fp, tn, fn = get_confusion_matrix(predictions, y_test)
        accuracy_list.append(accuracy)
        accuracy_matrix['tp'].append(tp)
        accuracy_matrix['fp'].append(fp)
        accuracy_matrix['tn'].append(tn)
        accuracy_matrix['fn'].append(fn)
    accuracy_matrix = {k: sum(v)/len(v) for k, v in accuracy_matrix.iteritems()}
    return sum(accuracy_list)/k, accuracy_matrix


def get_stem_plot(mle_list):
    df = pd.DataFrame(mle_list)

    markerline, stemlines, baseline = plt.stem(df[1], '-')
    plt.setp(baseline, 'color', 'b', 'linewidth', 2)
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(stemlines, 'color', 'b')

    markerline, stemlines, baseline = plt.stem(df[0], '-..')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.setp(markerline, 'markerfacecolor', 'r')
    plt.setp(stemlines, 'color', 'r')


    plt.legend(["Class 0 (ham)", "Class 1 (spam)"])
    plt.title("Bernoulli parameters for each class average accross 10 runs")
    plt.xlabel("Features")
    plt.ylabel("Bernoulli Params")
    plt.show()


def get_confusion_matrix(y_pred, y_test):
    df = pd.DataFrame([list(y_pred), list(y_test)]).transpose()
    total_correct = len(df[df[0] == df[1]])
    tp = len(df[(df[0] == 1) & (df[1] == 1)])
    fp = len(df[(df[0] == 1) & (df[1] == 0)])
    tn = len(df[(df[0] == 0) & (df[1] == 0)])
    fn = len(df[(df[0] == 0) & (df[1] == 1)])
    return float(total_correct) / len(y_pred), tp, fp, tn, fn


if __name__ == '__main__':
    df = get_df(scale=True, scale_by=100)
    X_train, y_train, X_test, y_test = split_data(df)
    mle_list = get_mle(X_train, y_train)
    get_stem_plot(mle_list)

    accuracy, accuracy_matrix = nb_classifier()
    print accuracy, accuracy_matrix

