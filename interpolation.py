import numpy as np

def lagrange(x, y, x_int):
    """
    Interpolates a value using Lagrange polynomial
    Parameters:
        x: Array containing x values
        y: Array containing y values
        x_int: Value to interpolate
    Returns:
        y_int: Interpolated value
    """

    n = x.size
    y_int = 0

    for i in range(0, n):
        p = y[i]
        for j in range(0, n):
            if i != j:
                p = p * (x_int - x[j]) / (x[i] - x[j])

                print('L{}(x) = (x - x_{}) / (x_{} - x_{}) '.format(i, j, i, j))
        y_int = y_int + p

    return [y_int]


def neville(x, y, x_int):
    """
    Interpolates a value using Neville polynomial
    Parameters:
        x: Array containing x values
        y: Array containing y values
        x_int: Value to interpolate
    Returns:
        y_int: Interpolated value
        q: Coefficients matrix
    """

    n = x.size
    q = np.zeros((n, n - 1))
    q = np.concatenate((y[:, None], q), axis=1)

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = ((x_int - x[i - j]) * q[i, j - 1] - (x_int - x[i]) * q[i - 1, j - 1]) / (x[i] - x[i - j])
            
    y_int = q[n - 1, n - 1]
    
    return [y_int, q]