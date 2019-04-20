import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt


def get_movies(fn):
    f = open(fn).readlines()
    movies = np.array([i.strip() for i in f])
    return movies


def get_matrix_shape(fn):
    """
    :param fn: (str) ratings_train
    :return: 
        num_movies: 1682
        num_users: 943
    """
    data = read_file(fn)
    num_movies = int(max([i[1] for i in data]))
    num_users = int(max([i[0] for i in data]))
    return num_movies, num_users


def init_matrix(n, m):
    """ Initialize """
    # create an empty matrix according to the dimensions of the training set.
    matrix = np.empty((n, m))
    matrix.fill(np.nan)
    return matrix


def get_matrix(fn, num_movies, num_users):
    """ Create an n by m matrix using movies and users.
    :param fn: (str) file name
    :param num_movies: (int) number of movies. should be 1682
    :param num_users: (int) number of users. should be 943
    """
    matrix = init_matrix(num_users, num_movies)
    f = open(fn).readlines()
    # add whatever values are available to the matrix, leave the rest nan
    for row in f:
        x, y, value = row.rstrip().split(',')
        x, y, value = int(x), int(y), float(value)
        # python is indexed at 0; need to adjust
        matrix[x - 1, y - 1] = value
    return matrix


def init_data(d, num_users, num_movies, lam):
    V = init_matrix(num_users, d)
    U = np.random.multivariate_normal(np.repeat(0, d), np.identity(d) / lam, num_movies).T
    return U, V


def squared_error(M, V, U):

    predicted = np.dot(V, U)
    rated_movies = ~np.isnan(M)

    error = ((M[rated_movies] - predicted[rated_movies]) ** 2).sum()
    return error


def squared_sum(matrix):
    return (matrix ** 2).sum()


def run(M, U, V, num_users, num_movies, iterations, lam, var, d):

    objectives = []
    for _ in range(iterations):
        for user_index in range(num_users):

            # all rated movies by user i.e. all movies per user that are not null
            rated_movies = ~np.isnan(M[user_index, :])

            # update a single row in R_i with the ratings for movies the user watched.
            U_i = U[:, rated_movies]

            # select all ratings for all movies rated for single user.
            matrix_train_i = M[user_index, rated_movies]

            V[user_index, :] = get_log_likelihood(U_i, lam, var, d, matrix_train_i)

        for movie_index in range(num_movies):

            # all not NaN ratings for a single movie
            rated_movies = ~np.isnan(M[:, movie_index])

            # update a single column for all users that rated this single movie
            V_j = V[rated_movies, :]

            # select all ratings for a single movie across all users
            matrix_train_j = M[rated_movies, movie_index]

            U[:, movie_index] = get_log_likelihood(V_j.T, lam, var, d, matrix_train_j)

        map_objective = get_map_objective(M, V, U, var, lam)
        objectives.append(map_objective)
    return objectives


def get_log_likelihood(U, lam, var, d, matrix_train):
    return np.linalg.inv(np.dot(U, U.T) + lam * var * np.identity(d)).dot(U.dot(matrix_train.T))


def get_map_objective(M, V, U, var, lam):
    return -(squared_error(M, V, U) / (2 * var) + squared_sum(V) * lam / 2 + squared_sum(U) * lam / 2)


def run_matrix_factorization(ratings_train_file, ratings_test_file, movies_file, d=10, lam=1, var=0.25, iterations=100):

    num_movies, num_users = get_matrix_shape(ratings_train_file)

    matrix_train = get_matrix(ratings_train_file, num_movies, num_users)
    matrix_test = get_matrix(ratings_test_file, num_movies, num_users)
    movies = get_movies(movies_file)

    # total number of rating across train and test
    num_ratings = np.count_nonzero(~np.isnan(matrix_train)) + np.count_nonzero(~np.isnan(matrix_test))

    filter_movies = ['Star Wars (1977)', 'My Fair Lady (1964)', 'GoodFellas (1990)']

    max_objective = -np.inf
    results = {}
    top_movies = None
    for iteration in range(d):
        print iteration
        U, V = init_data(d, num_users, num_movies, lam)

        objectives = run(matrix_train, U, V, num_users, num_movies, iterations, lam, var, d)

        min_objective = objectives[-1]
        results[iteration] = {
            'objectives': objectives,
            'min_objective': min_objective,
            'rmse_test': np.sqrt(squared_error(matrix_test, V, U) / num_ratings)
        }

        if min_objective > max_objective:
            # if we get an objective value that's lower than what we had before, update our findings.
            max_objective = min_objective
            top_movies = get_top_movies(U, filter_movies, movies)

    return results, top_movies


def plot_objective(results, iterations):
    plt.figure()
    colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'purple', 'pink', 'royalblue', 'sienna', 'orange']
    for k, v in results.iteritems():
        # for each run plot iteration 2 to 100
        plt.plot(range(2, iterations), v['objectives'][2:], colors[k])
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('Objective function for 10 runs')
    plt.legend(['run: %s' % (i + 1) for i in results])
    plt.show()
    plt.savefig('hw3_q3a_objectives.png')


def problem3a(results, iterations):
    plot_objective(results, iterations)
    results = pd.DataFrame(results).T[['min_objective', 'rmse_test']].sort_values('min_objective', ascending=False)
    results.to_csv('hw3_q3a_rmse.csv', index=False)


def problem3b(top_movies):
    top_movies.to_csv('hw3_q3b_top_movies.csv', index=False)


def get_top_movies(U, filter_movies, movies):
    top_movies = pd.DataFrame()
    for movie in filter_movies:

        # get index for movie
        movie_index = list(movies).index(movie)

        # get distance
        distances = np.round(np.sqrt(((U - U[:, movie_index].reshape(-1, 1)) ** 2).sum(axis=0)), 2)

        top_10_movie_ids = np.argsort(distances)[1:11]

        top_movies[movie] = zip(movies[top_10_movie_ids], distances[top_10_movie_ids])
    return top_movies


def read_file(file_name):
    """
    :param file_name: (str)
    :return:
        Read data from filename and return it as np.array()
    """
    fn = open(file_name, 'r')
    result = []
    for line in fn:
        record = line.strip()
        record_float = [float(c) for c in record.split(',')]
        result.append(record_float)
    fn.close()
    return np.array(result)


if __name__ == '__main__':
    ratings_train_file = "hw3-data/Prob3_ratings.csv"
    ratings_test_file = "hw3-data/Prob3_ratings_test.csv"
    movies_file = "hw3-data/Prob3_movies.txt"
    results, top_movies = run_matrix_factorization(ratings_train_file, ratings_test_file, movies_file)
    problem3a(results, iterations=100)
    problem3b(top_movies)
