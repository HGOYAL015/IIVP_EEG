import numpy as np

import scipy

import os

import sys

import scipy.signal

from sklearn.ensemble import RandomForestClassifier

import re

import nltk

from nltk.stem import *

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.io import loadmat

from PIL import _imaging

from sklearn.svm import SVC

from numpy import genfromtxt

from sklearn.datasets import make_classification

from sklearn.metrics import plot_confusion_matrix

import csv

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest

import pandas as pd

import numpy as np


def feature_mean(matrix):

    anss = np.mean(matrix, axis=0).flatten()

    naam = ['mean_' + str(i) for i in range(matrix.shape[1])]

    return anss, naam


def feature_mean_d(h1, h2):

    anss = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()

    naam = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]

    return anss, naam


def feature_mean_q(x_1, x_2, q3, q4):

    temp_1 = feature_mean(x_1)[0]

    temp_2 = feature_mean(x_2)[0]

    temp_3 = feature_mean(q3)[0]

    temp_4 = feature_mean(q4)[0]

    anss = np.hstack([
        temp_1, temp_2, temp_3, temp_4, temp_1 - temp_2, temp_1 - temp_3,
        temp_1 - temp_4, temp_2 - temp_3, temp_2 - temp_4, temp_3 - temp_4
    ]).flatten()

    naam = []

    for i in range(4):  # for all quarter-windows
        naam.extend(
            ['mean_q' + str(i + 1) + "_" + str(j) for j in range(len(temp_1))])

    for i in range(3):  # for quarter-windows 1-3

        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4

            naam.extend([
                'mean_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k)
                for k in range(len(temp_1))
            ])

    return anss, naam


def feature_stddev(matrix):

    anss = np.std(matrix, axis=0, ddof=1).flatten()

    naam = ['std_' + str(i) for i in range(matrix.shape[1])]

    return anss, naam


def feature_stddev_d(h1, h2):

    anss = (feature_stddev(h2)[0] - feature_stddev(h1)[0]).flatten()

    naam = ['std_d_h2h1_' + str(i) for i in range(h1.shape[1])]

    return anss, naam


def feature_moments(matrix):

    skw = scipy.stats.skew(matrix, axis=0, bias=False)

    krt = scipy.stats.kurtosis(matrix, axis=0, bias=False)

    anss = np.append(skw, krt)

    naam = ['skew_' + str(i) for i in range(matrix.shape[1])]

    naam.extend(['kurt_' + str(i) for i in range(matrix.shape[1])])

    return anss, naam


def feature_max(matrix):

    anss = np.max(matrix, axis=0).flatten()

    naam = ['max_' + str(i) for i in range(matrix.shape[1])]

    return anss, naam


def feature_max_d(h1, h2):

    anss = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()

    naam = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]

    return anss, naam


def feature_max_q(x_1, x_2, q3, q4):

    temp_1 = feature_max(x_1)[0]

    temp_2 = feature_max(x_2)[0]

    temp_3 = feature_max(q3)[0]

    temp_4 = feature_max(q4)[0]

    anss = np.hstack([
        temp_1, temp_2, temp_3, temp_4, temp_1 - temp_2, temp_1 - temp_3,
        temp_1 - temp_4, temp_2 - temp_3, temp_2 - temp_4, temp_3 - temp_4
    ]).flatten()

    naam = []

    for i in range(4):  # for all quarter-windows

        naam.extend(
            ['max_q' + str(i + 1) + "_" + str(j) for j in range(len(temp_1))])

    for i in range(3):  # for quarter-windows 1-3

        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4

            naam.extend([
                'max_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k)
                for k in range(len(temp_1))
            ])

    return anss, naam


def feature_min(matrix):

    anss = np.min(matrix, axis=0).flatten()
    naam = ['min_' + str(i) for i in range(matrix.shape[1])]
    return anss, naam


def feature_min_d(h1, h2):

    anss = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()

    naam = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return anss, naam


def feature_min_q(x_1, x_2, q3, q4):

    temp_1 = feature_min(x_1)[0]

    temp_2 = feature_min(x_2)[0]

    temp_3 = feature_min(q3)[0]

    temp_4 = feature_min(q4)[0]

    anss = np.hstack([
        temp_1, temp_2, temp_3, temp_4, temp_1 - temp_2, temp_1 - temp_3,
        temp_1 - temp_4, temp_2 - temp_3, temp_2 - temp_4, temp_3 - temp_4
    ]).flatten()

    naam = []

    for i in range(4):  # for all quarter-windows

        naam.extend(
            ['min_q' + str(i + 1) + "_" + str(j) for j in range(len(temp_1))])

    for i in range(3):  # for quarter-windows 1-3

        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4

            naam.extend([
                'min_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k)
                for k in range(len(temp_1))
            ])

    return anss, naam


def feature_covariance_matrix(matrix):

    covM = np.cov(matrix.T)

    indx = np.triu_indices(covM.shape[0])

    anss = covM[indx]

    naam = []

    for i in np.arange(0, covM.shape[1]):

        for j in np.arange(i, covM.shape[1]):

            naam.extend(['covM_' + str(i) + '_' + str(j)])

    return anss, naam, covM


def feature_eigenvalues(covM):

    anss = np.linalg.eigvals(covM).flatten()

    naam = ['eigenval_' + str(i) for i in range(covM.shape[0])]

    return anss, naam


def feature_logcov(covM):

    log_cov = scipy.linalg.logm(covM)

    indx = np.triu_indices(log_cov.shape[0])

    anss = np.abs(log_cov[indx])

    naam = []

    for i in np.arange(0, log_cov.shape[1]):

        for j in np.arange(i, log_cov.shape[1]):

            naam.extend(['logcovM_' + str(i) + '_' + str(j)])

    return anss, naam, log_cov


def feature_fft(matrix,
                period=1.,
                mains_f=50.,
                filter_mains=True,
                filter_DC=True,
                normalise_signals=True,
                ntop=10,
                get_power_spectrum=True):

    N = matrix.shape[0]

    T = period / N

    if normalise_signals:

        matrix = -1 + 2 * (matrix - np.min(matrix)) / \
            (np.max(matrix) - np.min(matrix))

    fft_values = np.abs(scipy.fft.fft(matrix, axis=0))[0:N // 2] * 2 / N

    freqs = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    if filter_DC:

        fft_values = fft_values[1:]
        freqs = freqs[1:]

    if filter_mains:

        indx = np.where(np.abs(freqs - mains_f) <= 1)

        fft_values = np.delete(fft_values, indx, axis=0)

        freqs = np.delete(freqs, indx)

    indx = np.argsort(fft_values, axis=0)[::-1]

    indx = indx[:ntop]

    anss = freqs[indx].flatten(order='F')

    naam = []

    for i in np.arange(fft_values.shape[1]):
        naam.extend(
            ['topFreq_' + str(j) + "_" + str(i) for j in np.arange(1, 11)])

    if (get_power_spectrum):

        anss = np.hstack([anss, fft_values.flatten(order='F')])

        for i in np.arange(fft_values.shape[1]):

            naam.extend([
                'freq_' + "{:03d}".format(int(j)) + "_" + str(i)
                for j in 10 * np.round(freqs, 1)
            ])

    return anss, naam


def calc_feature_vector(matrix, state):

    # Extract the half- and quarter-windows

    h1, h2 = np.split(matrix, [int(matrix.shape[0] / 2)])

    x_1, x_2, q3, q4 = np.split(matrix, [
        int(0.25 * matrix.shape[0]),
        int(0.50 * matrix.shape[0]),
        int(0.75 * matrix.shape[0])
    ])

    variablle_naam = []

    x, v = feature_mean(matrix)

    variablle_naam += v

    variablle_values = x

    x, v = feature_mean_d(h1, h2)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_mean_q(x_1, x_2, q3, q4)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_stddev(matrix)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_stddev_d(h1, h2)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_moments(matrix)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_max(matrix)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_max_d(h1, h2)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_max_q(x_1, x_2, q3, q4)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_min(matrix)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_min_d(h1, h2)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_min_q(x_1, x_2, q3, q4)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v, covM = feature_covariance_matrix(matrix)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_eigenvalues(covM)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v, _ = feature_logcov(covM)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    x, v = feature_fft(matrix)

    variablle_naam += v

    variablle_values = np.hstack([variablle_values, x])

    if state != None:

        variablle_values = np.hstack([variablle_values, np.array([state])])

        variablle_naam += ['Label']


#     print((variablle_naam))

    return variablle_values, variablle_naam


def generate_feature_vectors_from_samples(file_path,
                                          nsamples,
                                          period,
                                          state=None,
                                          remove_redundant=True,
                                          cols_to_ignore=None):
    csv_data = np.genfromtxt(file_path, delimiter=',')
    matrix = csv_data[1:]

    t = 0.

    previous_vector_ans = None

    anss = None

    while True:

        try:

            full_matrix = matrix
            start = t

            rstart = full_matrix[0, 0] + start

            index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))

            index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))

            # getting time slice

            duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]

            s = full_matrix[index_0:index_1, :]

            dur = duration

            if cols_to_ignore is not None:

                s = np.delete(s, cols_to_ignore, axis=1)
        except IndexError:
            break
        if len(s) == 0:
            break
        if dur < 0.9 * period:
            break

        ry, _ = scipy.signal.resample(s[:, 1:],
                                      num=nsamples,
                                      t=s[:, 0],
                                      axis=0)
        # print(ry)

        t += 0.5 * period

        r, headers = calc_feature_vector(ry, state)

        if previous_vector_ans is not None:

            feature_vector = np.hstack([previous_vector_ans, r])

            if anss is None:
                anss = feature_vector
            else:
                anss = np.vstack([anss, feature_vector])

        previous_vector_ans = r

        if state is not None:
            previous_vector_ans = previous_vector_ans[:-1]

    feat_naam = ["prev_" + s for s in headers[:-1]] + headers

    if remove_redundant:

        # Remove redundant lag window features

        to_rm = [
            "prev_mean_q3_", "prev_mean_q4_", "prev_mean_d_q3q4_",
            "prev_max_q3_", "prev_max_q4_", "prev_max_d_q3q4_", "prev_min_q3_",
            "prev_min_q4_", "prev_min_d_q3q4_"
        ]

        # Remove redundancies

        for i in range(len(to_rm)):

            for j in range(ry.shape[1]):

                rm_str = to_rm[i] + str(j)

                idx = feat_naam.index(rm_str)

                feat_naam.pop(idx)

                anss = np.delete(anss, idx, axis=1)

    print(len(anss))

    print(len(feat_naam))

    return anss, feat_naam


def gen_training_matrix(path_for_data, get_output_in_file, cols_to_ignore):

    FINAL_MATRIX = None

    for x in os.listdir(path_for_data):

        if not x.lower().endswith('.csv'):
            continue

        if 'test' in x.lower():
            continue
        try:
            _, state, _ = x[:-4].split('-')
        except:
            print('Wrong file name', x)
            sys.exit(-1)

        if state.lower() == 'concentrating':

            state = 2.
        elif state.lower() == 'neutral':

            state = 1.

        elif state.lower() == 'relaxed':

            state = 0.

        else:

            print('Wrong file name', x)

            sys.exit(-1)

        print('Using file', x)

        full_file_path = path_for_data + '/' + x

        vectors, header = generate_feature_vectors_from_samples(
            file_path=full_file_path,
            nsamples=150,
            period=1.,
            state=state,
            remove_redundant=True,
            cols_to_ignore=cols_to_ignore)

        print('resulting vector shape for the file', vectors.shape)

        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])

    print('FINAL_MATRIX', FINAL_MATRIX.shape)

    np.random.shuffle(FINAL_MATRIX)

    np.savetxt(get_output_in_file,
               FINAL_MATRIX,
               delimiter=',',
               header=','.join(header),
               comments='')

    return None


path_for_data = 'dataset/original_data/'

print(path_for_data)

get_output_in_file = 'out.csv'

gen_training_matrix(path_for_data, get_output_in_file, cols_to_ignore=-1)


