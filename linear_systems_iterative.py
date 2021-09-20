import numpy as np  
from numpy import dtype, linalg

def jacobi(A, b, x0, tol, N):  
    """
    Jacobi method: solve Ax = b given an initial approximation x0
    Parameters:
        a: Matrix A from system Ax=b
        b: Array containing b values
        x0: Initial approximation of solution
        tol: Tolerance
        iter_max: Maximum number of iterations
    Returns:
        x: Solution of linear system
    """
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double')  
 
    n = np.shape(A)[0]  
    x = np.zeros(n)  
    it = 0

    while (it < N):  
        it += 1

        for i in np.arange(n):  
            x[i] = b[i]  
            for j in np.concatenate((np.arange(0, i), np.arange(i + 1, n))):
                x[i] -= A[i, j] * x0[j]
            x[i] /= A[i, i]  

        new_epsilon = np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf)

        if (new_epsilon < tol):  
            return x  

        x0 = np.copy(x)  

    raise NameError('Max. iterations exceeded')

def gauss_seidel(A, b, x0, tol, N): 
    """
    Gauss-Seidel method: solve Ax = b given an initial approximation x0
    Parameters:
        a: Matrix A from system Ax=b
        b: Array containing b values
        x0: Initial approximation of solution
        tol: Tolerance
        iter_max: Maximum number of iterations
    Returns:
        x: Solution of linear system
    """
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double')  
 
    n = np.shape(A)[0]  
    x = np.copy(x0)  
    it = 0  

    while (it < N):  
        it += 1
  
        for i in np.arange(n):  
            x[i] = b[i]  
            for j in np.concatenate((np.arange(0,i),np.arange(i + 1, n))):  
                x[i] -= A[i, j] * x[j]  
            x[i] /= A[i, i]  

        new_epsilon = np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf)
        
        if (new_epsilon < tol):  
            return x  

        x0 = np.copy(x)  
    return x