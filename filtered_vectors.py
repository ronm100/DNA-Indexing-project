import os
from pathlib import Path
import numpy as np

VECTOR_SERIALIZATION_THRESHOLD = 5000000

class FilteredVectors:
    def __init__(self, vec_size: int, hmplmr_size: int, padding: int, save_dir: Path):
        self.vec_size = vec_size
        self.hmplmr_size = hmplmr_size
        self.padding = int(padding)
        self.curr_hmplmr = list()
        self.vectors = list()
        self.save_dir = save_dir
        self.vec_count = 1 # * VECTOR_SERIALIZATION_THRESHOLD

    def update_saved_vecs(self, vec: list = None):
        if vec:
            self.vectors.append(np.pad(np.array(vec, dtype=np.int8), (0, self.padding), 'wrap'))
        if (len(self.vectors) == VECTOR_SERIALIZATION_THRESHOLD or vec is None):
            file_name = self.save_dir / f'filtered_vecs_{self.vec_size}_{self.vec_count}.npy'
            self.vec_count += 1
            with open(file_name, 'wb') as f:
                np.save(f, np.array(self.vectors))
            self.vectors = list()

    def generate_vectors(self):
        for i in range(4):
            self.__explore_node(node=i, curr_vec=[])
        self.update_saved_vecs()
        return self.vectors

    def __explore_node(self, node: int, curr_vec):
        old_hmplmr = list(self.curr_hmplmr)
        if node in self.curr_hmplmr or not len(self.curr_hmplmr):
            self.curr_hmplmr.append(node)
        else:
            self.curr_hmplmr = [node]
        if len(self.curr_hmplmr) == self.hmplmr_size:
            self.curr_hmplmr = old_hmplmr
            return
        curr_vec.append(node)
        if len(curr_vec) == self.vec_size:
            self.update_saved_vecs(curr_vec)
        else:
            for i in range(4):
                self.__explore_node(i, curr_vec)
        curr_vec.pop()
        self.curr_hmplmr = old_hmplmr
        del old_hmplmr
