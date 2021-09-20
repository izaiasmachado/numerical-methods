import numpy as np

def lower_matrix_elements(n):
    elements = []

    for i in range(n):
        for j in range(n):
            if (i < j):
                elements.append([i, j]) 

    return elements

def gauss_elimination(matrix):
    coefficients = []
    elements = lower_matrix_elements(len(matrix))

    for elementId in range(len(elements)):
        element = elements[elementId]
        a, b = element
        elem_1 = matrix[b][a]
        elem_2 = matrix[a][a]
        x = elem_1 / elem_2
        coefficients.append({ 'x': a, 'y': b, 'value': x })

        l_k1 = np.array(matrix[b])
        l_k = np.array(matrix[a]) * (-x)

        answer = l_k1 + l_k
        matrix[b] = list(answer)

    L = np.array(list(matrix))
    L.fill(0)
    np.fill_diagonal(L, 1)
    L = np.transpose(L)[:-1]
    
    U = np.transpose(np.transpose(np.array(matrix))[:-1])

    for coefficient in coefficients:
        a = coefficient['x']
        b = coefficient['y']
        value = coefficient['value']
        L[b][a] = value

    return matrix, L, U


def gauss_elimination_partial_pivoting(matrix):
    elements = lower_matrix_elements(len(matrix))
    last_pivot = -10e6
    for elementId in range(len(elements)):
        element = elements[elementId]
        a, b = element

        if (last_pivot != a):
            pivot = []
            pivot_posi = a

            for i in range(len(matrix)):
                pivot.append(-10e6)

            for i in range(a, len(matrix)):
                if (np.absolute(matrix[i][a]) > pivot[a] and i != a):
                    pivot = matrix[i]
                    pivot_posi = i


            if (pivot_posi != a):
                matrix[pivot_posi], matrix[a] = matrix[a], matrix[pivot_posi]

            last_pivot = a

        elem_1 = matrix[b][a]
        elem_2 = matrix[a][a]
        x = elem_1 / elem_2

        l_k1 = np.array(matrix[b])
        l_k = np.array(matrix[a]) * (-x)

        answer = l_k1 + l_k
        matrix[b] = list(answer)

    return matrix