# Guillaume Payeur
# 260929164
import numpy as np
from scipy.linalg import eig
from scipy.linalg import eigh_tridiagonal

# constants
me = 9.11e-31
mp = 1.67e-27

# computing A
mu = (me*mp)/(me+mp)
A = 2*mu/me

# Finding the size of the matrix to be created
a = 0.03
p_max = 120
n = int(p_max/a)

# creating 1/p operator
invp = np.eye(n)
for i in range(n):
    invp[i,i] = invp[i,i]*(1/(a*(i+1)))

# creating second derivative operator
dp2 = np.zeros((n,n))
for i in range(1,n-1):
    dp2[i,i-1] = 1/(a**2)
    dp2[i,i] = -2/(a**2)
    dp2[i,i+1] = 1/(a**2)
dp2[0,0] = -2/a**2
dp2[0,1] = 1/a**2
dp2[-1,-2] = 1/a**2
dp2[-1,-1] = -2/a**2

# creating operator for -A/p - dp^2
M = -A*invp -dp2

# Getting eigenvalue pattern
eigenvalues = eigh_tridiagonal(np.diag(M),np.ones((n-1))*M[0,1])[0]
eigenvalues = 1/(eigenvalues/eigenvalues[0])
print('eigenvalue pattern for 6 lowest eigenvalues:')
print(eigenvalues[0:6])
