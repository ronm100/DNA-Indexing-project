from itertools import product
from typing import Tuple
import numpy as np
import math
import galois
from filtered_vectors import FilteredVectors
from multiprocessing import Pool
from edlib import align
import time

GF = galois.GF(4)
HMPLMR_LEN = 5


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
        if data_len <= message_len - redundancy_len:
            return message_len, redundancy_len
    return None, None  # Shouldn't get here


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
    # returns A matrix
    parity_check_matrix = get_parity_check_matrix(message_len, redundancy_len)
    # Delete I cols to get A matrix
    idx_to_delete = []
    for col_idx in range(len(parity_check_matrix.T)):
        if sum(parity_check_matrix[:, col_idx]) == 1:
            idx_to_delete.append(col_idx)
    A_matrix_tr = GF(np.delete(parity_check_matrix, idx_to_delete, axis=1))
    A_matrix = -A_matrix_tr.T

    return A_matrix


def create_indices(k: int, save_code_book: bool = False):
    print('starting get_code_dimensions')
    message_len, redundancy_len = get_code_dimensions(k)
    total_data_len = message_len - redundancy_len
    print('starting FilteredVectors')
    filtered_vectors = FilteredVectors(vec_size=k, hmplmr_size=HMPLMR_LEN,
                                       padding=total_data_len - k, save_vectors=save_code_book).generate_vectors()
    print('starting gen mat')
    gen_matrix = get_generator_matrix(int(message_len), redundancy_len)
    all_codes = [np.concatenate([vec, np.matmul(GF(vec.transpose()), gen_matrix)]) for vec in filtered_vectors]
    filtered_codes = [code for code in all_codes if not has_bad_sequence(code)]
    # code_book = {vec: np.dot(vec.transpose(), gen_matrix) for vec in filtered_vectors}
    if save_code_book:
        with open(f'code_book_{message_len}.npy', 'wb') as f:
            np.save(f, np.array(filtered_codes))
    print('finish create indices')
    return filtered_codes


def calc_edit_dist(words_tuple):
    word1, word2, i, j = words_tuple
    word1 = str(word1).replace('[', '').replace(']', '').replace(' ', '')
    word2 = str(word2).replace('[', '').replace(']', '').replace(' ', '')
    return align(word1, word2)['editDistance'] < 3, i, j


def calc_edit_dist_man(words_tuple):
    word1, word2, i, j = words_tuple
    word1 = str(word1).replace('[', '').replace(']', '').replace(' ', '')
    word2 = str(word2).replace('[', '').replace(']', '').replace(' ', '')
    return levenshtein_with_limit(word1, word2) < 3, i, j


def get_edit_dist_matrix(code_list: list):
    word_tuples = list()
    for i in range(len(code_list)):
        for j in range(i + 1, len(code_list)):
            word_tuples.append((code_list[i], code_list[j], i, j))
    time_0 = time.time()
    print('starting Pool')
    with Pool() as p:
        results = list(p.map(calc_edit_dist, word_tuples))
    time_1 = time.time()
    # with Pool() as p:
    #     results2 = list(p.map(calc_edit_dist_man, word_tuples))
    time_2 = time.time()
    print(f'edlib took {time_1 - time_0}')
    print(f'levenshtein_with_limit took {time_2 - time_1}')
    # print(f'resulst == results2?  {results == results2}')


    dim = len(code_list)
    shape = dim, dim
    matrix = np.zeros(shape=shape, dtype=np.int8)
    for dist, i, j in results:
        matrix[i][j] = dist
    return matrix


def filter_codes_by_edit_dist(init_code_book, distance_matrix):
    # while np.any(distance_matrix):
    #     bad_row_indices = np.where(np.any(distance_matrix > 0, axis=1))
    bad_row_indices = list(np.where(np.any(distance_matrix > 0, axis=1))[0])
    if not bad_row_indices:
        return init_code_book

    # bad_words = [init_code_book[i] for i in list(bad_row_indices)]
    filtered_code_book = [init_code_book[i] for i in range(len(init_code_book)) if i not in bad_row_indices]
    return list(filtered_code_book)


if __name__ == '__main__':
    unfiltered_code_book = create_indices(k=18, save_code_book=True)
    distance_mat = get_edit_dist_matrix(unfiltered_code_book)
    filtered_code_book = filter_codes_by_edit_dist(unfiltered_code_book, distance_mat)
    print('done')
    print(f'shape {distance_mat.shape}')
    print(distance_mat)
    # filtered_vectors = FilteredVectors(vec_size=3, hmplmr_size=2).generate_vectors()
    # print(filtered_vectors)
