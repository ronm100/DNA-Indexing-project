

class FilteredVectors:
    def __init__(self, vec_size, hmplmr_size):
        self.vec_size = vec_size
        self.hmplmr_size = hmplmr_size
        self.vectors = list()

    def generate_vectors(self):
        for i in range(4):
            self.__explore_node(node=i, curr_vec=[], curr_hmplmr=[])
        return self.vectors

    def __explore_node(self, node: int, curr_vec, curr_hmplmr):
        old_hmplmr = list(curr_hmplmr)
        if node in curr_hmplmr or not len(curr_hmplmr):
            curr_hmplmr.append(node)
        else:
            curr_hmplmr = list()
        if len(curr_hmplmr) == self.hmplmr_size:
            return
        curr_vec.append(node)
        if len(curr_vec) == self.vec_size:
            self.vectors.append(curr_vec)
        else:
            for i in range(4):
                self.__explore_node(i, curr_vec, curr_hmplmr)
        curr_vec.pop()
        curr_hmplmr = old_hmplmr
        del old_hmplmr