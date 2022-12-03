from itertools import product
from typing import Tuple
import numpy as np
import math

HMPLMR_LEN = 5


def has_bad_sequence(l: Tuple) -> bool:
    l = list(l)
    for i in range(len(l) - HMPLMR_LEN + 1):
        if l[i: i + HMPLMR_LEN] == [0] * HMPLMR_LEN or l[i: i + HMPLMR_LEN] == [1] * HMPLMR_LEN or \
                                                        l[i: i + HMPLMR_LEN] == [2] * HMPLMR_LEN or \
                                                        l[i: i + HMPLMR_LEN] == [3] * HMPLMR_LEN:
            return True

def calc_parity_bits(data: np.array):
    n_parity = math.ceil(math.log2(len(data))) + 1
    message = list(range(len(data) + n_parity))
    redundancy = np.zeros(n_parity, dtype=int)

    k = 0
    for i in range(len(message)):
        if math.log2(i + 1) == math.ceil(math.log2(i + 1)): # power of 2
            message[i] = 0
        else:
            message[i] = data[k]
            k += 1

    k = 0
    for i in range(len(message)):
        if math.log2(i + 1) == math.ceil(math.log2(i + 1)): # power of 2
            message[i] = (4 - sum(message[j - 1] for j in range(1, len(message) + 1) if j & int(2 ** (math.log2(i + 1)))) % 4) % 4
            redundancy[k] = message[i]
            k += 1
    #
    # for parity in range(n_parity):
    #     redundancy[parity] = 4 - sum(data[i - 1] for i in range(1, len(data) + 1) if i % (2 ** (parity + 1)) % 4) % 4
    return redundancy

def get_generator_matrix(data_len):
    gen = np.eye(data_len)
    redundancy = np.apply_along_axis(calc_parity_bits, 1, gen)
    gen = np.concatenate((gen,redundancy), axis=1)
    return gen


def create_indices(k: int):
    vector_space = [0, 1, 2, 3]
    all_vectors = product(vector_space, repeat=k)
    filtered_vectors = [np.array(vec) for vec in all_vectors if not has_bad_sequence(vec)]
    gen_matrix = get_generator_matrix(data_len=k)
    filtered_codes = [np.dot(vec.transpose(), gen_matrix) for vec in filtered_vectors]
    # code_book = {vec: np.dot(vec.transpose(), gen_matrix) for vec in filtered_vectors}
    return filtered_codes


if __name__ == '__main__':
    create_indices(k=4)
    # get_generator_matrix(7,4)
    # x = calc_parity_bits(np.array([1,0,0,0]))
    a = 1