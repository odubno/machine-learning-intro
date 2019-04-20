import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from collections import defaultdict


def get_data():
    """
    500 observations from three Gaussians with weights, mean and covariances
    """
    size = 500
    weights = [0.2, 0.5, 0.3]
    cov = np.matrix([
        [1, 0],
        [0, 1]
    ])
    mean_1 = np.array([0, 0])
    mean_2 = np.array([3, 0])
    mean_3 = np.array([0, 3])
    data_1 = multivariate_normal(mean=mean_1, cov=cov, size=size)
    data_2 = multivariate_normal(mean=mean_2, cov=cov, size=size)
    data_3 = multivariate_normal(mean=mean_3, cov=cov, size=size)

    select = np.random.choice(range(3), size=500, p=weights)

    data = np.concatenate((data_1[select == 0], data_2[select == 1], data_3[select == 2]))

    return data


def create_centers(k, data):
    """ Create random centroids
    :param k: (int) number of centroids
    :param data: (list) a list of lists where the nested lists are points on a graph
                        i.e.
                        [[ 6.13228898e-01,  1.47163226e+00],
                        [-2.60959819e-01,  1.81813093e-01]]
    :return:
        a list of lists with x and y coordinates to represent new cluster centers
    """
    centers = []
    x = np.array(zip(*data)[0])
    y = np.array(zip(*data)[1])
    for i in range(k):
        random_x = random.uniform(x.min(), x.max())
        random_y = random.uniform(y.min(), y.max())
        centers.append([random_x, random_y])
    return np.array(centers)


def expectation(centers, data):
    """ E-Step: assign points to the nearest cluster center
    :param data: (list) a list of lists where the nested lists are points on a graph
                        i.e.
                        [[ 6.13228898e-01,  1.47163226e+00],
                        [-2.60959819e-01,  1.81813093e-01]]
    :return:
        Labeled L2 distance
    """
    clusters = []
    for row in data:
        squared_sum = np.sum((centers - row) ** 2, axis=1)
        closest = min(squared_sum)
        label_index = list(squared_sum).index(closest)
        clusters.append((label_index, squared_sum[label_index]))
    return np.array(clusters)


def select_cluster(label_index, labels, data):
    """
    Returns data where the label matches it's index
    """
    return data[labels == label_index]


def maximize(k, labels, data, centers):
    """E-Step: assign points to the nearest cluster center.
    :param labels: (list) list of labels for clusters i.e. [0, 1, 1, 0, 0]
    :param data: (list) a list of lists where the nested lists are points on a graph
                        i.e.
                        [[ 6.13228898e-01,  1.47163226e+00],
                        [-2.60959819e-01,  1.81813093e-01]]
    :param centers: (list) a list of lists where the nested lists are points on a graph
    :return:
        An update list of list where the centers have been updated according to the mean of the clusters.
    """
    for label_index in range(k):
        cluster = select_cluster(label_index, labels, data)
        cluster_mean = np.mean(cluster, axis=0)
        # update old center to the new center (using the mean of the cluster)
        centers[label_index] = cluster_mean
    return centers


def run(k, data, iterations):
    """
    :param k: (int) number of clusters
    :param data: (list) a list of lists where the nested lists are points on a graph
                        i.e.
                        [[ 6.13228898e-01,  1.47163226e+00],
                        [-2.60959819e-01,  1.81813093e-01]]
    :param iterations:
    :return:
        each repetition of the E-step and M-step will always result in a better estimate of the cluster characteristics
        return those estimates as objectives.
    """

    """ 1. Initialization """
    objective = defaultdict(list)
    centers = create_centers(k, data)

    for _ in range(iterations):

        """ 2. E-Step: Assign clusters; assign points to the nearest cluster center """
        clusters = expectation(centers, data)

        labels = np.array(zip(*clusters)[0])
        features = np.array(zip(*clusters)[1])
        objective['labels'].append(labels)
        objective['features_sum'].append(sum(features))

        """ 3. M-Step: reset center; set the cluster centers to the mean """
        centers = maximize(k, labels, data, centers)
    return objective


def plot_objective_features(objectives, iterations):
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm']
    for k, v in objectives.iteritems():
        plt.plot(range(1, iterations + 1), v['features_sum'], colors[k])
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title('K-means objective function for 20 iterations. \n K = 2, 3, 4, 5')
    plt.legend(['k: %d' % (i + 2) for i in objectives])
    plt.savefig('hw3_q1a.png')
    plt.show()


def plot_k_clusters(data, objectives, k):
    colors = ['b', 'g', 'r', 'c', 'm']
    colors_arr = [colors[int(x)] for x in objectives[k-2]['labels'][0]]
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=colors_arr)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot - K=%d' % k)
    plt.savefig('hw3_q1b_k%s.png' % k)
    plt.show()


def run_kmeans(data, iterations, K):
    """
    :param data:
    :param iterations: (int) Number of times run() will minimize the objective function; keep centering centroids,
    :param K: (list) list of k integers; each k is the number of clusters the function will look for.
    :return:
        K-means objective function for N number of iterations
    """
    kmeans_objectives = {}
    for index in range(len(K)):
        k = K[index]
        kmeans_objective = run(k, data, iterations)
        kmeans_objectives[index] = kmeans_objective
    return kmeans_objectives


if __name__ == '__main__':
    iterations = 20
    data = get_data()
    K = [2, 3, 4, 5]

    main_objectives = run_kmeans(data, iterations, K)

    plot_objective_features(main_objectives, iterations)
    plot_k_clusters(data, main_objectives, k=3)
    plot_k_clusters(data, main_objectives, k=5)
