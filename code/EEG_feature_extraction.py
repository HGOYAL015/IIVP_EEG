
import numpy as np
import scipy
import os
import sys
import scipy.signal


def matrix_from_csv_file(file_path):

    csv_data = np.genfromtxt(file_path, delimiter=',')
    full_matrix = csv_data[1:]

    return full_matrix


def get_time_slice(full_matrix, start=0., period=1.):

    rstart = full_matrix[0, 0] + start
    index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
    index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))

    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration


def feature_mean(matrix):

    ret = np.mean(matrix, axis=0).flatten()
    names = ['mean_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_mean_d(h1, h2):

    ret = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()

    names = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names


def feature_mean_q(q1, q2, q3, q4):

    v1 = feature_mean(q1)[0]
    v2 = feature_mean(q2)[0]
    v3 = feature_mean(q3)[0]
    v4 = feature_mean(q4)[0]
    ret = np.hstack([v1, v2, v3, v4,
                     v1 - v2, v1 - v3, v1 - v4,
                     v2 - v3, v2 - v4, v3 - v4]).flatten()

    names = []
    for i in range(4):  # for all quarter-windows
        names.extend(['mean_q' + str(i + 1) + "_" + str(j)
                      for j in range(len(v1))])

    for i in range(3):  # for quarter-windows 1-3
        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4
            names.extend(['mean_d_q' + str(i + 1) + 'q' +
                          str(j + 1) + "_" + str(k) for k in range(len(v1))])

    return ret, names


def feature_stddev(matrix):

    ret = np.std(matrix, axis=0, ddof=1).flatten()
    names = ['std_' + str(i) for i in range(matrix.shape[1])]

    return ret, names


def feature_stddev_d(h1, h2):

    ret = (feature_stddev(h2)[0] - feature_stddev(h1)[0]).flatten()

    names = ['std_d_h2h1_' + str(i) for i in range(h1.shape[1])]

    return ret, names


def feature_moments(matrix):

    skw = scipy.stats.skew(matrix, axis=0, bias=False)
    krt = scipy.stats.kurtosis(matrix, axis=0, bias=False)
    ret = np.append(skw, krt)

    names = ['skew_' + str(i) for i in range(matrix.shape[1])]
    names.extend(['kurt_' + str(i) for i in range(matrix.shape[1])])
    return ret, names


def feature_max(matrix):

    ret = np.max(matrix, axis=0).flatten()
    names = ['max_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_max_d(h1, h2):

    ret = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()

    names = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names


def feature_max_q(q1, q2, q3, q4):

    v1 = feature_max(q1)[0]
    v2 = feature_max(q2)[0]
    v3 = feature_max(q3)[0]
    v4 = feature_max(q4)[0]
    ret = np.hstack([v1, v2, v3, v4,
                     v1 - v2, v1 - v3, v1 - v4,
                     v2 - v3, v2 - v4, v3 - v4]).flatten()

    names = []
    for i in range(4):  # for all quarter-windows
        names.extend(['max_q' + str(i + 1) + "_" + str(j)
                      for j in range(len(v1))])

    for i in range(3):  # for quarter-windows 1-3
        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4
            names.extend(['max_d_q' + str(i + 1) + 'q' +
                          str(j + 1) + "_" + str(k) for k in range(len(v1))])

    return ret, names


def feature_min(matrix):

    ret = np.min(matrix, axis=0).flatten()
    names = ['min_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_min_d(h1, h2):

    ret = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()

    names = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names


def feature_min_q(q1, q2, q3, q4):

    v1 = feature_min(q1)[0]
    v2 = feature_min(q2)[0]
    v3 = feature_min(q3)[0]
    v4 = feature_min(q4)[0]
    ret = np.hstack([v1, v2, v3, v4,
                     v1 - v2, v1 - v3, v1 - v4,
                     v2 - v3, v2 - v4, v3 - v4]).flatten()

    names = []
    for i in range(4):  # for all quarter-windows
        names.extend(['min_q' + str(i + 1) + "_" + str(j)
                      for j in range(len(v1))])

    for i in range(3):  # for quarter-windows 1-3
        for j in range((i + 1), 4):  # and quarter-windows (i+1)-4
            names.extend(['min_d_q' + str(i + 1) + 'q' +
                          str(j + 1) + "_" + str(k) for k in range(len(v1))])

    return ret, names


def feature_covariance_matrix(matrix):

    covM = np.cov(matrix.T)
    indx = np.triu_indices(covM.shape[0])
    ret = covM[indx]

    names = []
    for i in np.arange(0, covM.shape[1]):
        for j in np.arange(i, covM.shape[1]):
            names.extend(['covM_' + str(i) + '_' + str(j)])

    return ret, names, covM


def feature_eigenvalues(covM):

    ret = np.linalg.eigvals(covM).flatten()
    names = ['eigenval_' + str(i) for i in range(covM.shape[0])]
    return ret, names


def feature_logcov(covM):

    log_cov = scipy.linalg.logm(covM)
    indx = np.triu_indices(log_cov.shape[0])
    ret = np.abs(log_cov[indx])

    names = []
    for i in np.arange(0, log_cov.shape[1]):
        for j in np.arange(i, log_cov.shape[1]):
            names.extend(['logcovM_' + str(i) + '_' + str(j)])

    return ret, names, log_cov


def feature_fft(matrix, period=1., mains_f=50.,
                filter_mains=True, filter_DC=True,
                normalise_signals=True,
                ntop=10, get_power_spectrum=True):
    """
    Computes the FFT of each signal. 

    Parameters:
            matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
            values of nsignals for a time window of length nsamples
            period (float): width (in seconds) of the time window represented by
            matrix
            mains_f (float): the frequency of mains power supply, in Hz.
            filter_mains (bool): should the mains frequency (plus/minus 1Hz) be 
            filtered out?
            filter_DC (bool): should the DC component be removed?
            normalise_signals (bool): should the signals be normalised to the 
            before interval [-1, 1] before computing the FFT?
            ntop (int): how many of the "top N" most energetic frequencies should 
            also be returned (in terms of the value of the frequency, not the power)
            get_power_spectrum (bool): should the full power spectrum of each 
            signal be returned (in terms of magnitude of each frequency component)

    Returns:
            numpy.ndarray: 1D array containing the ntop highest-power frequencies 
            for each signal, plus (if get_power_spectrum is True) the magnitude of 
            each frequency component, for all signals.
            list: list containing feature names for the quantities calculated. The 
            names associated with the power spectrum indicate the frequencies down 
            to 1 decimal place.
    Author:
            Original: [fcampelo]
    """

    # Signal properties
    N = matrix.shape[0]  # number of samples
    T = period / N        # Sampling period

    # Scale all signals to interval [-1, 1] (if requested)
    if normalise_signals:
        matrix = -1 + 2 * (matrix - np.min(matrix)) / \
            (np.max(matrix) - np.min(matrix))

    # Compute the (absolute values of the) FFT
    # Extract only the first half of each FFT vector, since all the information
    # is contained there (by construction the FFT returns a symmetric vector).
    fft_values = np.abs(scipy.fft.fft(matrix, axis=0))[0:N//2] * 2 / N

    # Compute the corresponding frequencies of the FFT components
    freqs = np.linspace(0.0, 1.0 / (2.0 * T), N//2)

    # Remove DC component (if requested)
    if filter_DC:
        fft_values = fft_values[1:]
        freqs = freqs[1:]

    # Remove mains frequency component(s) (if requested)
    if filter_mains:
        indx = np.where(np.abs(freqs - mains_f) <= 1)
        fft_values = np.delete(fft_values, indx, axis=0)
        freqs = np.delete(freqs, indx)

    # Extract top N frequencies for each signal
    indx = np.argsort(fft_values, axis=0)[::-1]
    indx = indx[:ntop]

    ret = freqs[indx].flatten(order='F')

    # Make feature names
    names = []
    for i in np.arange(fft_values.shape[1]):
        names.extend(['topFreq_' + str(j) + "_" + str(i)
                      for j in np.arange(1, 11)])

    if (get_power_spectrum):
        ret = np.hstack([ret, fft_values.flatten(order='F')])

        for i in np.arange(fft_values.shape[1]):
            names.extend(['freq_' + "{:03d}".format(int(j)) + "_" + str(i)
                          for j in 10 * np.round(freqs, 1)])

    return ret, names


def calc_feature_vector(matrix, state):

    # Extract the half- and quarter-windows
    h1, h2 = np.split(matrix, [int(matrix.shape[0] / 2)])
    q1, q2, q3, q4 = np.split(matrix,
                              [int(0.25 * matrix.shape[0]),
                               int(0.50 * matrix.shape[0]),
                               int(0.75 * matrix.shape[0])])

    var_names = []

    x, v = feature_mean(matrix)
    var_names += v
    var_values = x

    x, v = feature_mean_d(h1, h2)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_mean_q(q1, q2, q3, q4)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_stddev(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_stddev_d(h1, h2)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_moments(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_max(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_max_d(h1, h2)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_max_q(q1, q2, q3, q4)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_min(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_min_d(h1, h2)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_min_q(q1, q2, q3, q4)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v, covM = feature_covariance_matrix(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_eigenvalues(covM)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v, _ = feature_logcov(covM)
    var_names += v
    var_values = np.hstack([var_values, x])

    x, v = feature_fft(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    if state != None:
        var_values = np.hstack([var_values, np.array([state])])
        var_names += ['Label']
#     print((var_names))
    return var_values, var_names


def generate_feature_vectors_from_samples(file_path, nsamples, period,
                                          state=None,
                                          remove_redundant=True,
                                          cols_to_ignore=None):
    csv_data = np.genfromtxt(file_path, delimiter=',')
    matrix = csv_data[1:]

    t = 0.

    previous_vector = None

    ret = None

    while True:

        try:
            s, dur = get_time_slice(matrix, start=t, period=period)
            if cols_to_ignore is not None:
                s = np.delete(s, cols_to_ignore, axis=1)
        except IndexError:
            break
        if len(s) == 0:
            break
        if dur < 0.9 * period:
            break

        ry, _ = scipy.signal.resample(s[:, 1:], num=nsamples,
                                      t=s[:, 0], axis=0)
        # print(ry)

        t += 0.5 * period

        r, headers = calc_feature_vector(ry, state)

        if previous_vector is not None:

            feature_vector = np.hstack([previous_vector, r])

            if ret is None:
                ret = feature_vector
            else:
                ret = np.vstack([ret, feature_vector])

        previous_vector = r
        if state is not None:
            previous_vector = previous_vector[:-1]

    feat_names = ["prev_" + s for s in headers[:-1]] + headers
    if remove_redundant:
        # Remove redundant lag window features
        to_rm = ["prev_mean_q3_", "prev_mean_q4_", "prev_mean_d_q3q4_",
                 "prev_max_q3_", "prev_max_q4_", "prev_max_d_q3q4_",
                 "prev_min_q3_", "prev_min_q4_", "prev_min_d_q3q4_"]

        # Remove redundancies
        for i in range(len(to_rm)):
            for j in range(ry.shape[1]):
                rm_str = to_rm[i] + str(j)
                idx = feat_names.index(rm_str)
                feat_names.pop(idx)
                ret = np.delete(ret, idx, axis=1)
    print(len(ret))
    print(len(feat_names))
    return ret, feat_names


def gen_training_matrix(directory_path, output_file, cols_to_ignore):

    FINAL_MATRIX = None

    for x in os.listdir(directory_path):

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
        full_file_path = directory_path + '/' + x
        vectors, header = generate_feature_vectors_from_samples(file_path=full_file_path,
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

    np.savetxt(output_file, FINAL_MATRIX, delimiter=',',
               header=','.join(header),
               comments='')

    return None


directory_path = 'dataset/original_data/'
print(directory_path)
output_file = 'out.csv'
gen_training_matrix(directory_path, output_file, cols_to_ignore=-1)
