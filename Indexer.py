from itertools import product
from pathlib import Path
import pickle
from typing import Tuple
import numpy as np
import os
import galois
from filtered_vectors import FilteredVectors
from multiprocessing import Pool
from edlib import align

GF = galois.GF(4)
HMPLMR_LEN = 5

def get_code_dimensions(data_len: int) -> Tuple[int, int]:
    for redundancy_len in range(data_len):
        message_len = ((4 ** redundancy_len) - 1) / 3
        if data_len == message_len - redundancy_len:
            return message_len, redundancy_len
    raise ValueError('data_len is probably too big')

def get_parity_check_matrix(message_len: int, redundancy_len: int) -> np.array:
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


def get_generator_matrix(message_len: int, redundancy_len: int) -> np.array:
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

def generate_vecs_without_homopolymers(k: int, raw_vec_dir: Path):
    message_len, redundancy_len = get_code_dimensions(k)
    total_data_len = message_len - redundancy_len
    FilteredVectors(vec_size=k, hmplmr_size=HMPLMR_LEN,
                                       padding=total_data_len - k, save_dir=raw_vec_dir).generate_vectors()

def generate_hamming_codes(k: int, raw_vec_dir: Path, hamming_code_dir: Path):
    message_len, redundancy_len = get_code_dimensions(k)

    gen_matrix = get_generator_matrix(int(message_len), redundancy_len)
    for filename in os.listdir(raw_vec_dir):
        filtered_vectors = np.load(raw_vec_dir / filename)
        all_codes = np.concatenate((filtered_vectors, np.matmul(GF(filtered_vectors), gen_matrix)), axis=1)
        # a filter that checks for homopolymers created by concatanating the message to the redundancy
        filter = np.concatenate((np.expand_dims(np.amin(all_codes[:,k-4:k+1], axis=1) == np.amax(all_codes[:,k-4:k+1], axis=1),axis=1),
                            np.expand_dims(np.amin(all_codes[:,k-3:k+2], axis=1) == np.amax(all_codes[:,k-3:k+2], axis=1),axis=1),
                            np.expand_dims(np.amin(all_codes[:,k-2:k+3], axis=1) == np.amax(all_codes[:,k-2:k+3], axis=1),axis=1)), axis=1)
        filter = np.any(filter, axis=1)
        filtered_codes = all_codes[~filter].astype(np.int8)
        np.save(hamming_code_dir / filename, filtered_codes)


def calc_edit_dist(words_tuple) -> Tuple[int,int,int]:
    word1 = str(words_tuple[:21]).replace('[','').replace(',','').replace(']','').replace(' ','')
    word2 = str(words_tuple[21:]).replace('[','').replace(',','').replace(']','').replace(' ','')
    return align(word1, word2)['editDistance'] < 3, word1, word2


def calc_distances(codes: np.array, edit_dists_dir: Path):
    n_rows = codes.shape[0]
    word_tuples = np.zeros(shape=(1, 2*codes.shape[1]),dtype=np.int8)
    j = 0

    for i in range(1, int(n_rows/2)):
        words = np.concatenate((codes, np.roll(codes, shift=i, axis=0)), axis=1)
        word_tuples = np.concatenate((word_tuples,words),axis=0)
        if i % int(n_rows / 100) == 0:
            with Pool() as p:
                results = list(p.map(calc_edit_dist, word_tuples[1:].tolist()))
            with open(edit_dists_dir / f'{j}.pkl','wb') as f:
                pickle.dump(results, f)
            word_tuples = np.zeros(shape=(1, 2*codes.shape[1]),dtype=np.int8)
            j += 1

def index_codes(codes: np.array, edit_dists_dir: Path):
    word_to_num = {str(codes[i,:]).replace('[','').replace(',','').replace(']','').replace(' ',''):i for i in range(len(codes))}
    num_to_word = {i : str(codes[i,:]).replace('[','').replace(',','').replace(']','').replace(' ','') for i in range(len(codes))}
    with open(edit_dists_dir / 'word_to_num.pkl', 'wb') as f:
        pickle.dump(word_to_num, f)
    with open(edit_dists_dir / 'num_to_word.pkl', 'wb') as f:
        pickle.dump(num_to_word, f)

def calc_edit_dist_matrix(edit_dists_dir: Path):
    with open(edit_dists_dir / 'word_to_num.pkl','rb') as f:
        word_to_num = pickle.load(f)

    dim = len(word_to_num)
    shape = dim, dim
    matrix = np.zeros(shape=shape, dtype=np.int8)

    for filename in os.listdir(edit_dists_dir):
        if 'dists_' not in filename:
            continue
        with open(edit_dists_dir / filename,'rb') as f:
            dists = pickle.load(f)
        dists = [dist for dist in dists if dist[0] == True]
        for dist in dists:
            matrix[word_to_num[dist[1]], word_to_num[dist[2]]] = 1

    np.save(edit_dists_dir / 'dist_matrix.npy', matrix)



def filter_codes_by_edit_dist(edit_dists_dir: Path):
    with open(edit_dists_dir / 'num_to_word.pkl', 'rb') as f:
        num_to_word = pickle.load(f)

    distance_matrix = np.load(edit_dists_dir / 'dist_matrix.npy')
    while np.any(distance_matrix):
        n_mismatch = np.sum(distance_matrix, axis=1)
        max_val_idx = np.argmax(n_mismatch)
        distance_matrix[max_val_idx,:] = np.zeros(shape=distance_matrix.shape[0], dtype=np.int8)
        distance_matrix[:,max_val_idx] = np.zeros(shape=distance_matrix.shape[0], dtype=np.int8)
        del num_to_word[max_val_idx]

    with open(edit_dists_dir / 'final_codes.pkl', 'wb') as f:
        pickle.dump(num_to_word, f)

if __name__ == '__main__':
    k = 18
    raw_vec_dir = Path('generated_vectors')
    hamming_code_dir = Path('hamming_codes')
    edit_dists_dir = Path('edit_dists')
    
    for dir in [raw_vec_dir, hamming_code_dir, edit_dists_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            raise ValueError(f'trying to override {dir}')
    poc_percentage = 1 / 100

    generate_vecs_without_homopolymers(k=k, raw_vec_dir=raw_vec_dir)
    generate_hamming_codes(k=k, raw_vec_dir=raw_vec_dir, hamming_code_dir=hamming_code_dir)

    filtered_code_book = np.load(hamming_code_dir / 'filtered_vecs_18_1.npy')
    frac = int(filtered_code_book.shape[0] * poc_percentage)
    filtered_code_book = filtered_code_book[0:frac,:]
    calc_distances(filtered_code_book, edit_dists_dir=edit_dists_dir)
    index_codes(filtered_code_book, edit_dists_dir=edit_dists_dir)
    calc_edit_dist_matrix(edit_dists_dir)
    filter_codes_by_edit_dist(edit_dists_dir)
