import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from collections import defaultdict


def expectation(data, cov, mu, weight, normalize_cov):
    """
    :param data: (numpy.ndarray) nxm matrix
    :param cov: (numpy.ndarray) covariances for each cluster
    :param mu: (numpy.ndarray) means for each cluster
    :param weight: (numpy.ndarray) probability for each cluster
    :param normalize_cov: (numpy.ndarray) prevents singularity matrix errors when calculating covariance matrices
    :return:
        r_ic gives us for each data point x_i the measure of:
            Probability that x_i belongs to class c divided by
            Probability of x_i over all classes
        Hence if x_i is very close to one gaussian c,
        it will get a high r_ic value for this gaussian and relatively low values otherwise.
    """
    r_ic = np.zeros((len(data), len(cov)))
    for m, co, p, r in zip(mu, cov, weight, range(len(r_ic[0]))):
        co += normalize_cov
        mn = multivariate_normal(mean=m, cov=co)

        # probability that x belongs to gaussian c in column c.
        prob_x_belongs_to_class_c = p * mn.pdf(data)

        prob_of_all_classes = get_prob_of_all_classes(data, weight, mu, cov, normalize_cov)

        prob = prob_x_belongs_to_class_c / prob_of_all_classes
        # sometimes probabilities are equal to 0. Dividing zero by zero causes nan.
        # replacing such values with 0.0
        prob[np.isnan(prob)] = 0.0
        r_ic[:, r] = prob
    return r_ic


def get_prob_of_all_classes(data, weight, mu, cov, normalize_cov):
    prob = []
    for weight_c, mu_c, cov_c in zip(weight, mu, cov + normalize_cov):
        prob.append(weight_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(data))
    return np.sum(prob, axis=0)


def maximize(data, normalize_cov, r_ic):
    mu = []
    cov = []
    weight = []
    for c in range(len(r_ic[0])):
        cluster_c = r_ic[:, c]

        # fraction of points allocated to cluster c
        m_c = np.sum(cluster_c, axis=0)

        # probability that a point belongs to cluster c (r_ic)
        mu_c = np.sum(data * cluster_c.reshape(len(data), 1), axis=0) / m_c
        mu.append(mu_c)

        # Calculate the covariance matrix for each cluster based on the new mean
        cov_update = np.dot((np.array(cluster_c).reshape(len(data), 1) * (data - mu_c)).T, (data - mu_c)) / m_c
        con_norm = cov_update + normalize_cov
        cov.append(con_norm)

        # probability assigned to each source
        m = np.sum(r_ic)
        weight_c = m_c / m
        weight.append(weight_c)

    return mu, cov, weight


def get_cov_matrix(data, k):
    # We need a nxmxm covariance matrix for each source
    # since we have m features --> create symmetric covariance matrices with ones on the diagonal
    cov = np.zeros((k, len(data[0]), len(data[0])))
    for dim in range(len(cov)):
        np.fill_diagonal(cov[dim], 1)
    return cov


def get_mu(k, data):
    mu = []
    data_min = min(data[:, 0])
    data_max = max(data[:, 0])
    for _ in range(k):
        mu.append([random.uniform(data_min, data_max) for _ in range(len(data[0]))])
    return np.array(mu)


def initialize_values(data, k):
    # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
    mu = get_mu(k, data)

    # nxmxm covariance matrix for each source
    cov = get_cov_matrix(data, k)

    # Are "Fractions"
    weight = np.ones(k) / k

    return mu, cov, weight


def run(k, data, iterations):
    """
    :param k: (int)
    :param data: (numpy.ndarray)
    :param iterations: (int)
    """

    """ 1. Initialize; Set the initial mu, covariance and weight values"""
    mu, cov, weight = initialize_values(data, k)

    # helps prevent cov matrix from becoming zero by offsetting it by an arbitrary value
    normalize_cov = 1e-8 * np.identity(len(data[0]))
    objectives = defaultdict(list)
    for index in range(iterations):

        """ 2. E-Step: Get r_ic """
        # Calculate the probability of each data point x_i belonging to cluster c
        r_ic = expectation(data, cov, mu, weight, normalize_cov)

        """ 3. M-Step: Update expectation """
        # Calculate the new mean vector and new covariance matrices
        mu, cov, weight = maximize(data, normalize_cov, r_ic)

        # Log likelihood
        objectives['log_likelihood'].append(get_log_likelihood(data, mu, cov, weight))
        print '%s iteration' % index

    # saving the best run of 30 iterations
    objectives['mu'].append(mu)
    objectives['cov'].append(cov)
    objectives['weight'].append(weight)
    return objectives


def get_log_likelihood(data, mu, cov, weight):
    likelihood = []
    for weight_i, mu_i, cov_i in zip(weight, range(len(mu)), range(len(cov))):
        result = weight_i * multivariate_normal(mu[mu_i], cov[cov_i]).pdf(data)
        likelihood.append(result)
    return np.log(np.sum(likelihood))


def run_gmm(data, iterations, K):
    """
    :param data: (numpy.ndarray)
    :param iterations: (int) Number of times run() will minimize the objective function; keep centering centroids,
    :param K: (list) list of k integers; each k is the number of clusters the function will look for.
    :return:
        K-means objective function for N number of iterations
    """
    gmm_objectives = {}
    index = 0
    while True:
        k = K[index]
        print '%s initialization with k=%s' % (index, k)
        objective = run(k, data, iterations)
        gmm_objectives[index] = objective
        index += 1
        if len(gmm_objectives) == len(K):
            break
    return gmm_objectives


def read_file(file_name):
    """
    :param file_name: (str)
    :return:
        Read data from filename and return it as np.array()
    """
    fn = open(file_name, 'r')
    result = []
    # read each remaining feature and convert each one to float.
    for line in fn:
        record = line.strip()
        record_float = [float(c) for c in record.split(',')]
        result.append(record_float)
    fn.close()
    return np.array(result)


def split_by_class(X, Y):
    X_class_0 = np.array([x for x, y in zip(X, Y) if y[0] == 0])
    X_class_1 = np.array([x for x, y in zip(X, Y) if y[0] == 1])
    return [X_class_0, X_class_1]


def plot_objective_features(objectives, iterations, c):
    plt.figure()
    colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'purple', 'pink', 'royalblue', 'sienna', 'orange']
    for k, v in objectives.iteritems():
        # plot iteration 5 to 30
        plt.plot(range(5, iterations+1), v['log_likelihood'][4:], colors[k])
    plt.xlabel('Iterations')
    plt.ylabel('Log Marginal Objective')
    plt.title('10 initializations with k=3 and for iterations 5 to 30 for class %s' % c)
    plt.legend(['k: %d' % (i + 1) for i in objectives])
    plt.savefig('hw3_q2a_class%s.png' % c)
    plt.show()


def problem2a():
    # Problem 2a
    # 10 models with k=3 over 30 iterations for each class
    K = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    iterations = 30
    X_train = read_file('hw3-data/Prob2_Xtrain.csv')
    Y_train = read_file('hw3-data/Prob2_Ytrain.csv')
    for index, data in enumerate(split_by_class(X_train, Y_train)):
        main_objectives = run_gmm(data, iterations, K)
        plot_objective_features(main_objectives, iterations, c=index)


def get_pdf(X, objectives, k):
    pdfs = np.zeros((len(X), len(objectives['weight'][0])))
    for index in range(len(X)):
        row = X[index]
        for k_i in range(k):
            weight = objectives['weight'][0][k_i]
            mean = objectives['mu'][0][k_i]
            cov = objectives['cov'][0][k_i]
            pdfs[index, k_i] = weight * multivariate_normal(mean=mean, cov=cov).pdf(row)
    return np.sum(pdfs, axis=1)


def problem2b():
    K = [1, 2, 3, 4]
    iterations = 30
    X_train = read_file('hw3-data/Prob2_Xtrain.csv')
    Y_train = read_file('hw3-data/Prob2_Ytrain.csv')

    X_test = read_file('hw3-data/Prob2_Xtest.csv')
    y_test = read_file('hw3-data/Prob2_ytest.csv')
    y_test = [int(i[0]) for i in y_test]

    X_train_0, X_train_1 = split_by_class(X_train, Y_train)
    main_objectives_0 = run_gmm(X_train_0, iterations, K)
    main_objectives_1 = run_gmm(X_train_1, iterations, K)

    for index in range(len(K)):
        k = K[index]
        pdf_0 = get_pdf(X_test, main_objectives_0[index], k)
        pdf_1 = get_pdf(X_test,  main_objectives_1[index], k)
        y_pred = [0 if i < 0 else 1 for i in np.log(pdf_1 / pdf_0)]
        r = get_confusion_matrix(y_pred, y_test)
        print 'k=%s |' % K[index], r


def get_confusion_matrix(y_pred, y_test):
    df = pd.DataFrame([list(y_pred), list(y_test)]).transpose()
    total_correct = len(df[df[0] == df[1]])
    tp = len(df[(df[0] == 1) & (df[1] == 1)])
    fp = len(df[(df[0] == 1) & (df[1] == 0)])
    tn = len(df[(df[0] == 0) & (df[1] == 0)])
    fn = len(df[(df[0] == 0) & (df[1] == 1)])
    accuracy = float(total_correct) / len(y_pred)
    return {'accuracy': accuracy, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def prediction_accuracy(y_pred, y_test):
    total_predictions = len(y_pred)
    correct_predictions = [i for i in range(total_predictions) if y_pred[i] == y_test[i]]
    return len(correct_predictions) / float(total_predictions)


if __name__ == '__main__':
    problem2a()
    problem2b()
