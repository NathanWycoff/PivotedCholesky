#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  pivoted_chol.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.14.2020

## A pivoted cholesky function for kernel functions
import numpy as np

def pivoted_chol(get_diag, get_row, M, err_tol = 1e-6):
    """
    A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive semi-definite operator. Only diagonal elements and select rows of that operator's matrix represenation are required.

    get_diag - A function which takes no arguments and returns the diagonal of the matrix when called.
    get_row - A function which takes 1 integer argument and returns the desired row (zero indexed).
    M - The maximum rank of the approximate decomposition; an integer. 
    err_tol - The maximum error tolerance, that is difference between the approximate decomposition and true matrix, allowed. Note that this is in the Trace norm, not the spectral or frobenius norm. 

    Returns: R, an upper triangular matrix of column dimension equal to the target matrix. It's row dimension will be at most M, but may be less if the termination condition was acceptably low error rather than max iters reached.
    """

    d = np.copy(get_diag())
    N = len(d)

    pi = list(range(N))

    R = np.zeros([M,N])

    err = np.sum(np.abs(d))

    m = 0
    while (m < M) and (err > err_tol):

        i = m + np.argmax([d[pi[j]] for j in range(m,N)])

        tmp = pi[m]
        pi[m] = pi[i]
        pi[i] = tmp

        R[m,pi[m]] = np.sqrt(d[pi[m]])
        Apim = get_row(pi[m])
        for i in range(m+1, N):
            if m > 0:
                ip = np.inner(R[:m,pi[m]], R[:m,pi[i]])
            else:
                ip = 0
            R[m,pi[i]] = (Apim[pi[i]] - ip) / R[m,pi[m]]
            d[pi[i]] -= pow(R[m,pi[i]],2)

        err = np.sum([d[pi[i]] for i in range(m+1,N)])
        m += 1

    R = R[:m,:]

    return(R)

# Example:
N = 10
S = np.random.normal(size=[N,N])
A = np.matmul(S.T, S)

# Important to copy as diag may not be writable depending on np version
get_diag = lambda: np.diag(A).copy()
get_row = lambda i: A[i,:]
R = pivoted_chol(get_diag, get_row, M = N)

# R.T @ R is numerically equal to A.
print(np.linalg.norm(np.matmul(R.T, R) - A))
