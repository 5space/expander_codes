import numpy as np
from core import *
import networkx as nx
import networkx.generators.expanders as nxg
import scipy as sp
import random

print("Expander Graphs")

def generator_matrix(matrix):
    k, n = matrix.shape
    # reduced form
    aug = np.hstack((matrix.T, np.eye(n, dtype=int)))

    free_vars = []
    for i in range(k):
        for j in range(i, n):
            if aug[j, i] == 1:
                aug[[i, j]] = aug[[j, i]]
                break
        else:
            free_vars.append(i)
            continue
        
        # sweep out the column
        for j in range(n):
            if j != i and aug[j, i] == 1:
                aug[j] ^= aug[i]
    
    free_vars.extend(range(k, n))
    gen = aug[free_vars, k:].T
    return gen

def make_expander_code(n, c, d, max_iters=1000):

    assert (c*n) % d == 0, "n*c must be divisible by d"

    edges = np.random.permutation(np.arange(c*n))
    edges_pairs = set()

    constraint_neighbors = [[] for _ in range(n*c//d)]
    variable_neighbors = [[] for _ in range(n)]
    
    for i, j in enumerate(edges):
        Ci, Vj = i//d, j//c
        edges_pairs.add((Ci, Vj))
        constraint_neighbors[Ci].append(Vj)
        variable_neighbors[Vj].append(Ci)

    row, col = zip(*edges_pairs)
    data_row, data_col = np.array(row), np.array(col)
    data = np.ones(len(data_row), dtype=int)
    # initialize new sparse matrix
    A = sp.sparse.csc_matrix((data, (data_row, data_col)), shape=(n*c//d, n))
    gen = generator_matrix(A.todense())

    def encoder(plaintext: np.ndarray) -> np.ndarray:
        return (gen @ plaintext) % 2

    def decoder(ciphertext: np.ndarray) -> np.ndarray:
        guess = ciphertext.copy()

        # A.shape = (constraints, variables)

        bins = [set() for _ in range(c+1)]
        bins_key = [0] * A.shape[1]
        unsatisfied = (A @ guess) % 2
        for Vj in range(A.shape[1]):
            unsatisfied_neighbors = np.sum(unsatisfied[variable_neighbors[Vj]])
            bins[unsatisfied_neighbors].add(Vj)
            bins_key[Vj] = unsatisfied_neighbors
        
        for _ in range(max_iters):

            # find highest occupied bin
            max_uns = c
            while bins[max_uns] == set():
                max_uns -= 1
            
            if 2*max_uns <= c:
                return guess
            
            # find variable with most unsatisfied neighbors
            v = random.choice(tuple(bins[max_uns]))
            guess[v] ^= 1

            for Ci in variable_neighbors[v]:
                unsatisfied[Ci] ^= 1
                if unsatisfied[Ci] == 1:
                    d_uns = 1
                else:
                    d_uns = -1
                
                for Vj in constraint_neighbors[Ci]:
                    bins[bins_key[Vj]].remove(Vj)
                    bins_key[Vj] += d_uns
                    bins[bins_key[Vj]].add(Vj)
    
    return gen.shape[1], encoder, decoder

block_size = 2000
n, enc, dec = make_expander_code(block_size, 5, 10, max_iters=2000)

# print(run_code(enc, noisy_channel_fixed_errors, dec, 0.02, n, 100))

xs = []
ys = []
for errs in range(1, 61):
    print(errs)
    xs.append(errs)
    ys.append(run_code(enc, noisy_channel_fixed_errors, dec, errs/n, n, 50))
    

from matplotlib import pyplot as plt

plt.plot(xs, ys)
plt.xlabel("Errors")
plt.ylabel("Success Rate")
plt.title("(2000, 5, 10) Expander Code Success Rate")
plt.grid()
plt.savefig("expander.png")
plt.show()