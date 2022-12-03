from itertools import product
from typing import Tuple

HMPLMR_LEN = 5

def has_bad_sequence(l: Tuple) -> bool:
    l = list(l)
    for i in range(len(l) - HMPLMR_LEN + 1):
        if l[i : i+HMPLMR_LEN] == [0] * HMPLMR_LEN or l[i : i+HMPLMR_LEN] == [1] * HMPLMR_LEN or l[i : i+HMPLMR_LEN] == [2] * HMPLMR_LEN or l[i : i+HMPLMR_LEN] == [3] * HMPLMR_LEN:
            return True

def create_indices(k: int):
    vector_space = [0, 1, 2 ,3]
    all_vectors = product(vector_space, repeat=k)
    filtered_vectors = [vec for vec in all_vectors if not has_bad_sequence(vec)]
    return filtered_vectors

if __name__ == '__main__':
    create_indices(k=7)