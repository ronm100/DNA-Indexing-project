from itertools import product
from pathlib import Path
import pickle
import tracemalloc
from typing import Tuple
import numpy as np
import os
import math
import galois
from filtered_vectors import FilteredVectors, VECTOR_SERIALIZATION_THRESHOLD
from multiprocessing import Pool
from edlib import align
import time

GF = galois.GF(4)
HMPLMR_LEN = 5
VECS_PER_FILE = 50000


def levenshtein_with_limit(str1, str2, limit=3):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j == n and min(dp[i]) >= limit:
                return limit
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def has_bad_sequence(vec: Tuple) -> bool:
    vec = list(vec)
    for i in range(len(vec) - HMPLMR_LEN + 1):
        if vec[i: i + HMPLMR_LEN] == [0] * HMPLMR_LEN or vec[i: i + HMPLMR_LEN] == [1] * HMPLMR_LEN or \
                vec[i: i + HMPLMR_LEN] == [2] * HMPLMR_LEN or \
                vec[i: i + HMPLMR_LEN] == [3] * HMPLMR_LEN:
            return True


def get_code_dimensions(data_len: int) -> Tuple[int, int]:
    for redundancy_len in range(data_len):
        message_len = ((4 ** redundancy_len) - 1) / 3
        if data_len == message_len - redundancy_len:
            return message_len, redundancy_len
    raise ValueError('data_len is probably too big')


def get_parity_check_matrix(message_len, redundancy_len):
    total_len = 1
    curr_block_exp = 1
    vector_space = [0, 1, 2, 3]
    parity_check_matrix = np.transpose(np.array((redundancy_len - 1) * [0] + [1]))
    parity_check_matrix = np.expand_dims(parity_check_matrix, axis=1)
    while total_len < message_len:
        block = np.zeros((redundancy_len - curr_block_exp - 1, 4 ** curr_block_exp))
        block = np.concatenate([block, np.ones((1, 4 ** curr_block_exp))], axis=0)
        vec_combination = [vec for vec in product(vector_space, repeat=curr_block_exp)]
        block = np.concatenate([block, np.transpose(np.array(vec_combination))], axis=0)
        parity_check_matrix = np.concatenate([parity_check_matrix, block], axis=1)
        total_len += 4 ** curr_block_exp
        curr_block_exp += 1
    return parity_check_matrix.astype(int)


def get_generator_matrix(message_len, redundancy_len):
    # returns 'A' matrix
    parity_check_matrix = get_parity_check_matrix(message_len, redundancy_len)
    # Delete 'I' cols to get 'A' matrix
    idx_to_delete = []
    for col_idx in range(len(parity_check_matrix.T)):
        if sum(parity_check_matrix[:, col_idx]) == 1:
            idx_to_delete.append(col_idx)
    A_matrix_tr = GF(np.delete(parity_check_matrix, idx_to_delete, axis=1))
    A_matrix = -A_matrix_tr.T

    return A_matrix


def generate_codes(k: int):
    print('starting get_code_dimensions')
    message_len, redundancy_len = get_code_dimensions(k)
    # total_data_len = message_len - redundancy_len
    # print('starting FilteredVectors')
    # # filtered_vectors = FilteredVectors(vec_size=k, hmplmr_size=HMPLMR_LEN,
    # #                                    padding=total_data_len - k, save_vectors=save_code_book).generate_vectors()
    gen_matrix = get_generator_matrix(int(message_len), redundancy_len)
    for filename in os.listdir('generated_vectors610'):
        filtered_vectors = np.load(Path('generated_vectors610') / filename)
        all_codes = np.concatenate((filtered_vectors, np.matmul(GF(filtered_vectors), gen_matrix)), axis=1)
        # a filter that checks for homopolymers created by concatanating the message to the redundancy
        filter = np.concatenate((np.expand_dims(np.amin(all_codes[:,14:19], axis=1) == np.amax(all_codes[:,14:19], axis=1),axis=1),
                            np.expand_dims(np.amin(all_codes[:,15:20], axis=1) == np.amax(all_codes[:,15:20], axis=1),axis=1),
                            np.expand_dims(np.amin(all_codes[:,16:21], axis=1) == np.amax(all_codes[:,16:21], axis=1),axis=1)), axis=1)
        filter = np.any(filter, axis=1)
        filtered_codes = all_codes[~filter].astype(np.int8)
        np.save(Path('generated_vectors610_filtered') / filename, filtered_codes)
        print(filename)
    print('finish create indices')


def calc_edit_dist(words_tuple):
    # word1, word2 = words_tuple
    word1 = str(words_tuple[:21]).replace('[','').replace(',','').replace(']','').replace(' ','')
    word2 = str(words_tuple[21:]).replace('[','').replace(',','').replace(']','').replace(' ','')
    return align(word1, word2)['editDistance'] < 3, word1, word2


def calc_distances(codes: np.array):
    n_rows = codes.shape[0]
    print(f'n_rows = {n_rows}')
    word_tuples = np.zeros(shape=(1, 2*codes.shape[1]),dtype=np.int8)
    print('start matrix calc')
    time_0 = time.time()
    tracemalloc.start()
    for i in range(1, int(n_rows/2)):
        words = np.concatenate((codes, np.roll(codes, shift=i, axis=0)), axis=1)
        word_tuples = np.concatenate((word_tuples,words),axis=0)
        # print(i)
        if i % int(n_rows / 100) == 0:
            print(i)
            print(tracemalloc.get_tracemalloc_memory())
            print(word_tuples.shape)
            with Pool() as p:
                results = list(p.map(calc_edit_dist, word_tuples[1:].tolist()))
            with open(f'edit_distances_01/{int(i / (n_rows / 100))}.pkl','wb') as f:
                pickle.dump(results, f)
            word_tuples = np.zeros(shape=(1, 2*codes.shape[1]),dtype=np.int8)
    print(f'word_tuples.shape = {word_tuples.shape}')


    time_1 = time.time()
    print(f'tuples took {time_1 - time_0}, starting Pool')
    time_2 = time.time()
    print(f'edlib took {time_2 - time_1}')

def index_codes(codes: np.array):
    word_to_num = {str(codes[i,:]).replace('[','').replace(',','').replace(']','').replace(' ',''):i for i in range(len(codes))}
    print('done_1')
    num_to_word = {i : str(codes[i,:]).replace('[','').replace(',','').replace(']','').replace(' ','') for i in range(len(codes))}
    print('done_2')
    with open(f'edit_distances_01/word_to_num.pkl','wb') as f:
        pickle.dump(word_to_num, f)
    with open(f'edit_distances_01/num_to_word.pkl','wb') as f:
        pickle.dump(num_to_word, f)

def calc_edit_dist_matrix(dists_dir):
    with open(f'edit_distances_01/word_to_num.pkl','rb') as f:
        word_to_num = pickle.load(f)
    # with open(f'edit_distances_01/num_to_word.pkl','rb') as f:
    #     num_to_word = pickle.load(f)

    dim = len(word_to_num)
    shape = dim, dim
    matrix = np.zeros(shape=shape, dtype=np.int8)
    for filename in os.listdir(dists_dir):
        print(f'start {filename}')
        if 'dists_' not in filename:
            continue
        with open(f'edit_distances_01/{filename}','rb') as f:
            dists = pickle.load(f)
        # try:
        dists = [dist for dist in dists if dist[0] == True]
        # except TypeError:
        #     print('passed')
        #     pass
        for dist in dists:
            matrix[word_to_num[dist[1]], word_to_num[dist[2]]] = 1
    np.save(Path('edit_distances_01') / 'dist_matrix.npy', matrix)



def filter_codes_by_edit_dist(distance_matrix):
    with open(f'edit_distances_01/num_to_word.pkl','rb') as f:
        num_to_word = pickle.load(f)
    while np.any(distance_matrix):
        # bad_row_indices = np.where(np.any(distance_matrix > 0, axis=1))
        n_mismatch = np.sum(distance_matrix, axis=1)
        max_val_idx = np.argmax(n_mismatch)
        distance_matrix[max_val_idx,:] = np.zeros(shape=distance_matrix.shape[0], dtype=np.int8)
        distance_matrix[:,max_val_idx] = np.zeros(shape=distance_matrix.shape[0], dtype=np.int8)
        del num_to_word[max_val_idx]
    print('done')
    with open(f'edit_distances_01/final_codes.pkl','wb') as f:
        pickle.dump(num_to_word, f)


if __name__ == '__main__':
    # generate_codes(k=18)

    # filtered_code_book = np.load('generated_vectors610_filtered/filtered_vecs_18_1.npy')
    # frac = int(filtered_code_book.shape[0] / 100)
    # filtered_code_book = filtered_code_book[0:frac,:]
    # calc_edit_dist_matrix('edit_distances_01')

    distance_matrix = np.load('edit_distances_01/dist_matrix.npy')
    filter_codes_by_edit_dist(distance_matrix)

    # filtered_code_book = filter_codes_by_edit_dist(unfiltered_code_book, distance_mat)
    # print('done')
    # print(f'shape {distance_mat.shape}')
    # print(distance_mat)
    # filtered_vectors = FilteredVectors(vec_size=3, hmplmr_size=2).generate_vectors()
    # print(filtered_vectors)




# cmd = /usr/bin/python3 -u /home_nfs/ronmaishlos/DNA-Indexing-project/Indexer.py 2>&1 | tee out.log