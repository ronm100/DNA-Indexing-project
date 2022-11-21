from itertools import product

def has_bad_sequence(l: list) -> bool:
    return ((0,) * 5 in l) or ((1,) * 5 in l) or ((2,) * 5 in l) or ((3,) * 5 in l) 

def create_indices(k: int):
    vector_space = [0, 1, 2 ,3]
    all_vectors = product(vector_space, repeat=k)

    # filtered_vectors = filter(has_bad_sequence, all_vectors)
    filtered_vectors = [vec for vec in all_vectors if not has_bad_sequence(vec)]
    all_vectors_2 = [vec for vec in all_vectors]
    print(len(list(all_vectors)))
    print(len(list(filtered_vectors)))

if __name__ == '__main__':
    create_indices(k=5)
    # print(has_bad_sequence((0,0,0,0,0)))