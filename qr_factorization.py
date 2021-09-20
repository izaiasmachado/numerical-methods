import math
import numpy as np

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

def convert_column(x):
    new_x = np.array(x)[np.newaxis]
    transposed = new_x.T
    return transposed

def calculateR(q, v): 
    transposed_column_v = convert_column(v)
    r = np.dot(q, transposed_column_v)[0]
    return r

def qr(matrix):
    a = []
    q = []
    transpost_matrix = np.transpose(matrix)
    r_matrix = np.array(list(transpost_matrix))
    r_matrix.fill(0)

    for vectorId in range(len(transpost_matrix)):
        vector = transpost_matrix[vectorId]

        value = vector
        for qId in range(len(q)):
            r = calculateR(q[qId], vector)
            r_matrix[qId][vectorId] = r
            value -= r * q[qId]

        a.append(value)
        norm_a = norm(a[vectorId])
        q.append(a[vectorId] / norm_a)
        r_matrix[vectorId][vectorId] = calculateR(q[vectorId], vector)

    return np.transpose(q), r_matrix

def qr_method(a, k):
    for i in range(k):
        print('K = {}'.format(i))

        q, r = qr(a)
        a = np.dot(r, q)

        print('Q = ')
        pp(q)
        print('R = ')
        pp(r)
        print('A = ')
        pp(a)