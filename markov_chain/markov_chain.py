import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy.linalg as nplg
from scipy.sparse.linalg import eigs


def get_matrix(M_shape=767):
    """ M is  transition matrix, or Markov matrix.
    :param M_shape: (int)
    :return:
        a transition matrix describing the probabilities of particular transitions.
    """
    # initialize matrix
    scores_file = 'hw4_data/CFB2018_scores.csv'
    M = np.zeros((M_shape, M_shape))
    data = open(scores_file, 'r').readlines()
    for row in data:
        # define index and points for each team
        a_index, a_points, b_index, b_points = [int(i) for i in row.split(',')]

        # define which team won
        a_wins = int(a_points > b_points)
        b_wins = int(a_points < b_points)

        # get average by team
        total_points = float(a_points + b_points)
        a_weight = a_points / total_points
        b_weight = b_points / total_points

        # define i and j indices
        # entry (i, j) is the probability of transitioning from state i to state j
        i = a_index - 1
        j = b_index - 1

        # team ranking; map all the values
        M[i][i] = a_wins + a_weight
        M[j][i] = a_wins + a_weight
        M[j][j] = b_wins + b_weight
        M[i][j] = b_wins + b_weight

    # normalize each row
    # each row must add up to exactly 1 because each row represents its own probability distribution.
    # count how many times a transition from i to j is observed and divide by the total number of transitions from i.
    for row_index in xrange(M_shape):
        M[row_index] = M[row_index] / np.sum(M[row_index])

    return M


def problem_1a(M, M_shape=767):
    """
        Rank the teams by sorting in decreasing value according to this vector.
        List the top 25 team names and their corresponding values in w_t for t = 10, 100, 1000, 10000.
    :param M: (numpy.ndarray)
    :param M_shape: (int)
    :return:
        Transitions only occur between teams that play each other.
        Transitions happen from teams that lose to teams that win.
        If Team A beats Team B, there should be a high probability of transitioning from B to A and small probability from A to B.
        The strength of the transition is linked to the score of the game.
    """
    w_t = []
    for t in [10, 100, 1000, 10000]:
        w = (1.0 / M_shape) * np.ones(M_shape)
        for i in xrange(t):
            w = np.dot(w, M)
        w_t.append(w)
    top_teams = get_top_teams(w_t)
    top_teams.to_csv('hw4_q1a.csv', index=False)


def problem_1b(M, M_shape=767):
    """ Plot |w_t - w_inf|  as a function of t for t=1,...,10000
    :param M: (numpy.ndarray)
    :param M_shape: (int)
\    """
    w_t = []
    w = (1.0 / M_shape) * np.ones(M_shape)
    for i in xrange(10000):
        w = np.dot(w, M.T)
        w_t.append(w)
    data = get_eigen_vectors(M, w_t)
    plot_norms(data)


def get_eigen_vectors(M, w_t):
    w_inf = eigs(M.T, 1)[1].flatten().T
    w_inf = w_inf / np.sum(w_inf)
    data = nplg.norm(w_t - w_inf, 1, axis=1)
    return data


def get_teams():
    f = open('hw4_data/TeamNames.txt', 'r')
    data = f.readlines()
    return {i: v.strip() for i, v in enumerate(data)}


def get_top_teams(w_t):
    teams_map = get_teams()
    t_w_map = {
        0: 't=10',
        1: 't=100',
        2: 't=1000',
        3: 't=10000'
    }
    df = pd.DataFrame()
    for t in range(len(w_t)):
        top_25_team_index = np.argsort(w_t[t])[::-1][:25]
        top_25_teams = [teams_map[i] for i in top_25_team_index]
        top_25_weights = w_t[t][top_25_team_index]
        df[t_w_map[t]] = zip(top_25_teams, top_25_weights)
    return df


def plot_norms(data):
    plt.figure()
    plt.plot(range(10000), data)
    plt.title("||w_t - w_inf|| for t=1,...,10000")
    plt.xlabel("t")
    plt.ylabel("||w_t - w_inf||")
    plt.savefig('hw4_q1b.png')
    plt.show()


if __name__ == '__main__':
    M = get_matrix()
    problem_1a(M)
    problem_1b(M)


