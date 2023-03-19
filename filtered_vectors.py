import numpy as np

# Debug imports:
from pympler import tracker, asizeof
import time 

class FilteredVectors:
    def __init__(self, vec_size, hmplmr_size, padding, save_vectors = False):
        self.vec_size = vec_size
        self.hmplmr_size = hmplmr_size
        self.padding = int(padding)
        self.curr_hmplmr = list()
        self.vectors = list()
        self.save_vectors = save_vectors
        self.vec_count = 1 # * 10000000

    def update_saved_vecs(self, vec: list = None):
        if vec:
            self.vectors.append(np.pad(np.array(vec, dtype=np.int8), (0, self.padding), 'wrap'))
        if (len(self.vectors) == 10000000 or vec is None) and self.save_vectors:
            file_name = f'./generated_vectors/filtered_vecs_{self.vec_size}_{self.vec_count}.npy'
            self.vec_count += 1
            print(f'vec count = {self.vec_count}')
            with open(file_name, 'wb') as f:
                np.save(f, np.array(self.vectors))
            self.vectors = list()

    def generate_vectors(self):
        self.time_0 = time.time()
        self.tr = tracker.SummaryTracker()
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
            
            if (time.time() - self.time_0) >= 10 * 60:
                self.time_0 = time.time()
                vecs_sz = asizeof.asizeof(self.vectors)
                print(f'size in MB: {vecs_sz / (2 ** 20)}, size in GB: {vecs_sz / (2 ** 30)} with {len(self.vectors)} vectors')
                self.tr.print_diff()
                self.tr = tracker.SummaryTracker()
        else:
            for i in range(4):
                self.__explore_node(i, curr_vec)
        curr_vec.pop()
        self.curr_hmplmr = old_hmplmr
        del old_hmplmr
