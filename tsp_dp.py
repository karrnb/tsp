import itertools
import random
import sys
import numpy as np

def generate_matrix(dim):    
    mat = np.random.random_integers(0,100,size=(dim,dim))
    # mat = [[0] * dim for i in range(dim)]
    np.fill_diagonal(mat, 0)
    b_symm = (mat + mat.T)/2
    return b_symm

# Implementation of Held_Karp to solve Travelling Salesman Problem using 
# Dynamic Programming
# Algorithm from (https://en.wikipedia.org/wiki/Held-Karp_algorithm)
# We take each node, generate subsets for each of them along with their
# weights and then calculate the minmum, backtracking the path for each
# takes matrix and dimension of matrix as input
# gives us minimum cost and a path to solving TSP
def tsp(mat, dim):
    # Map combinations of all subsets to the cost required to reach this 
    # subset along with path
    # All values stored as left shifted binary bits and their corresponding 
    # decimal values
    Map = {}

    # subset in bits
    for i in range(1, dim):
        # Initializing
        Map[(1 << i, i)] = (mat[0][i], 0)

    for i in range(2,dim):
        # taking all possible nodes accessible from i as subset
        for comboSubset in itertools.combinations(range(1,dim), i):
            # we set all the bits in each subset to map all the nodes for this
            # all the nodes are concated onto bits
            bits = 0
            for x in comboSubset:
                bits = bits | 1 << x

            for j in comboSubset:
                # we left shift j and take it's complement to find the previous 
                # prevbit is the binary string of previous bits of j which we 
                # use to reference in Map[] to find the par
                # prevbit = bits & ((1 << j) - 1)
                prevbit = bits & ~(1 << j)

                # initialize Out to store the paths of all the nodes to i
                # this is our DAG
                Out = []
                for k in comboSubset:
                    if k == 0 or k == j:
                        # skip because diagnol or symmetric other
                        continue
                    # store each mapping of parent and node in Out
                    Out.append((Map[(prevbit, k)][0] + mat[k][j], k))
                # the minimum two of all nodes is mapped to subset for 
                # each subset with total cost to that node
                Map[(bits, j)] = min(Out)

    # removing (0,0) start node
    bits = (2**dim - 1) - 1

    # once the mapping has been obtained, we find the minimum cost
    MinimumCost = []
    for i in range(1, dim):
        MinimumCost.append((Map[(bits, i)][0] + mat[i][0], i))
    cost, parentNode = min(MinimumCost)

    # finding path to printsimply storing each node and finding 
    # it's parent and storing the same
    path = []
    for i in range(dim - 1):
        # path stores the nodes traversed in reverse order
        path.append(parentNode)
        # we backtrack from the parentNode to find it's parent
        nb = bits & ~(1 << parentNode)
        # replace the parent of the parentNode
        x, parentNode = Map[(bits), parentNode]
        bits = nb

    return cost, path

def main():
    # generate a symmetric matrix as input graph
    # taking input dimension from user.
    matrix = generate_matrix(int(sys.argv[1]))
    print(matrix)

    dim = len(matrix)
    cost, path = tsp(matrix, dim)
    # adding zero for start and reversing order
    path.append(0).reverse()
    # printing output
    print('Cost: ' + str(cost))
    print('Path: ' + str(path))


if __name__ == '__main__':
    main()