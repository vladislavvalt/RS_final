__author__ = 'vladvalt'

import numpy as np
import random
import pandas as pd
from scipy import spatial

# 0 values should be treated as unknown
grades = [0, 0, 0, 1, 2, 3, 4, 5]
# number of neighbours to use for filtering
num_neighbours = 2
initial_matrix_location = "../input_data/initial.csv"
recommendation_matrix_location = "../output_data/recommendation.csv"


def read_matrix(path):
    initial_matrix_frame = pd.read_csv(path, header=None)
    initial_nd_array = pd.DataFrame.as_matrix(initial_matrix_frame)
    initial_matrix = np.matrix(initial_nd_array, dtype=float)
    return initial_matrix


def save_matrix(matrix, path):
    """
    directory must exist
    :param matrix:
    :param path:
    :return:
    """
    matrix_frame = pd.DataFrame(matrix)
    matrix_frame.to_csv(path_or_buf=path, header=False, index=False, float_format="%.2f")


def generate_matrix(path, n, m):
    initial_arr = [[random.choice(grades) for j in range(m)] for i in range(n)]
    initial_matrix = np.matrix(initial_arr, dtype=float)
    save_matrix(initial_matrix, path)


def collaborative_filtering(r_matrix):
    if isinstance(r_matrix, np.matrix):

        def normalize_r_matrix():
            for i in range(n):
                row_means[i] = float(r_matrix[i, :].sum()) / m

            w_matrix = r_matrix.copy()
            if isinstance(w_matrix, np.matrix):
                for i in range(n):
                    for j in range(m):
                        if w_matrix[i, j] != 0:
                            w_matrix[i, j] -= row_means[i]
            return w_matrix

        def fill_recommendations(rec_matrix):
            recommendations = r_matrix.copy()
            for i in range(n):
                for j in range(m):
                    if r_matrix[i, j] == 0:
                        recommendations[i, j] += rec_matrix[i, j] + row_means[i]
            return recommendations

        def build_correlation_matrix(some_matrix):
            correlation_matrix = np.identity(n, dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    corr = spatial.distance.cosine(some_matrix[i, :], some_matrix[j, :])
                    correlation_matrix[i, j] = correlation_matrix[j, i] = corr
            return correlation_matrix

        def predict_ratings():
            for i in range(n):
                correlation_vector = correlation_matrix[i, :]
                # correlation_vector = np.delete(correlation_vector, correlation_vector.argmax())
                for j in range(m):
                    if n_matrix[i, j] == 0:
                        neighbours = []
                        required_neighbours_num = num_neighbours
                        while required_neighbours_num > 0 and len(correlation_vector) > 0:
                            k = correlation_vector.argmax()
                            if n_matrix[k, j] != 0:
                                neighbours.append(n_matrix[k, j])
                                required_neighbours_num -= 1
                            correlation_vector = np.delete(correlation_vector, k)

                        if len(neighbours) > 0:
                            n_matrix[i, j] += float(sum(neighbours)) / len(neighbours)

        n, m = r_matrix.shape
        row_means = [0.0 for i in range(n)]

        n_matrix = normalize_r_matrix()

        correlation_matrix = build_correlation_matrix(n_matrix)

        predict_ratings()

        recommendation_matrix = fill_recommendations(n_matrix)
        return recommendation_matrix

# uncomment to try recommender on another matrix
# generate_matrix(initial_matrix_location, 10, 15)

print "Read matrix from", initial_matrix_location
r = read_matrix(initial_matrix_location)
print "Initial rating matrix:"
print r
print
print

recommendation_matrix = collaborative_filtering(r)

print "Recommendation matrix build:"
print recommendation_matrix
print
print

print "Saving matrix to", recommendation_matrix_location
save_matrix(recommendation_matrix, recommendation_matrix_location)


