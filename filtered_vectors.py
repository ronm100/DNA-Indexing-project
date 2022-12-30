import numpy as np

class FilteredVectors:
    def __init__(self, vec_size, hmplmr_size, padding):
        self.vec_size = vec_size
        self.hmplmr_size = hmplmr_size
        self.padding = int(padding)
        self.curr_hmplmr = list()
        self.vectors = list()

    def generate_vectors(self):
        for i in range(4):
            self.__explore_node(node=i, curr_vec=[])
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
            self.vectors.append(np.pad(np.array(curr_vec), (0, self.padding), 'constant', constant_values=(1, 1)))
        else:
            for i in range(4):
                self.__explore_node(i, curr_vec)
        curr_vec.pop()
        self.curr_hmplmr = old_hmplmr
        del old_hmplmr