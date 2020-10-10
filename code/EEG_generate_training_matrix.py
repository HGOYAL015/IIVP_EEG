

import os
import sys
import numpy as np
from EEG_feature_extraction import generate_feature_vectors_from_samples


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


if __name__ == '__main__':

    directory_path = 'dataset/original_data/'
    print(directory_path)
    output_file = 'out.csv'
    gen_training_matrix(directory_path, output_file, cols_to_ignore=-1)
