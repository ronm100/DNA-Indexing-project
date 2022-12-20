from itertools import product
from typing import Tuple
import numpy as np
import math
import galois

HMPLMR_LEN = 5


def has_bad_sequence(l: Tuple) -> bool:
    l = list(l)
    for i in range(len(l) - HMPLMR_LEN + 1):
        if l[i: i + HMPLMR_LEN] == [0] * HMPLMR_LEN or l[i: i + HMPLMR_LEN] == [1] * HMPLMR_LEN or \
                l[i: i + HMPLMR_LEN] == [2] * HMPLMR_LEN or \
                l[i: i + HMPLMR_LEN] == [3] * HMPLMR_LEN:
            return True


def get_code_dimensions(data_len: int) -> (int, int):
    for i in range(100):
        if data_len <= (4 ** i) - 1 - i:
            return (4 ** i) - 1, (4 ** i) - 1 - i
    return None, None  # Shouldnt get here

def get_generator_matrix(data_len):
    message_len, total_data_len = get_code_dimensions(data_len)
    parity_len = message_len - total_data_len
    non_zero_vecs = product([0, 1], repeat=parity_len)
    A_trans = [vec for vec in non_zero_vecs if sum(vec) > 1]
    A_trans = GF(A_trans)
    A_mat = np.transpose(-A_trans)
    return A_mat


def create_indices(k: int):
    vector_space = [0, 1, 2, 3]
    all_vectors = product(vector_space, repeat=k)
    filtered_vectors = [np.array(vec) for vec in all_vectors if not has_bad_sequence(vec)]
    gen_matrix = get_generator_matrix(data_len=k)
    filtered_codes = [np.dot(vec.transpose(), gen_matrix) for vec in filtered_vectors]
    # code_book = {vec: np.dot(vec.transpose(), gen_matrix) for vec in filtered_vectors}
    return filtered_codes


if __name__ == '__main__':
    GF = galois.GF(4)
    create_indices(k=13)
    # get_generator_matrix(7,4)
    # x = calc_parity_bits(np.array([1,0,0,0]))
    a = 1
