import quimb as qu
from quspin.operators import hamiltonian, exp_op, quantum_operator
from quspin.tools.measurements import ent_entropy
from quspin.basis import spin_basis_1d
import numpy as np
import params
import h_functions as hf
from scipy.special import comb
from time import time
import sys,os
import math

LOCAL = os.path.abspath('.')
PATH_now = LOCAL

ti = time() # start timing function for each realization

LOCAL = os.path.abspath('.')
PATH_now = LOCAL

N = params.L
W_i = params.W_i
W = params.D
J = 1.0
dis_flag = params.Dis_gen
seed = int(1000000*np.random.random())
int_flag = params.Int_flag
t_tab = np.logspace(-1, 1.5, 200)

if int_flag==0:
    J_tab=(J,J,0)
else:
    J_tab=(J,J,J)

P = qu.zspin_projector(N, sz=0)

if dis_flag==1:
    H_0 = qu.ham_mbl(N, W_i, J_tab, cyclic=False, dh_dist='qp', beta=0.721, seed=seed, sparse=True).real
else:
    H_0 = qu.ham_mbl(N, W_i, J_tab, cyclic=False, seed=seed, sparse=True).real

H_pre = P.T @ H_0 @ P

psi_0 = qu.eigvecsh(H_pre, k=1, which='SA').ravel()

if dis_flag==1:
    H_1 = qu.ham_mbl(N, W, J_tab, cyclic=False, dh_dist='qp', beta=0.721, seed=seed, sparse=True).real
else:
    H_1 = qu.ham_mbl(N, W, J_tab, cyclic=False, seed=seed, sparse=True).real

H_post = P.T @ H_1 @ P

compute = {
    'time': lambda t, p: t,
    'losch': lambda t, p: qu.fidelity(psi_0, p)
}
evo = qu.Evolution(psi_0, H_post, compute=compute, method='expm')

<<<<<<< HEAD
for t in evo.at_times(t_tab):
    continue

TS=evo.results['time']
LOSCH=np.array(-np.log(evo.results['losch']))/N
=======
TS=[]
for t in evo.at_times(t_tab):
    TS.append(t)

LOSCH=np.array(-np.log(evo.results['losch']))/N

>>>>>>> 0f4baebdf5adf995e750780e77c6e22e00ce5969

if dis_flag == 1:
    directory = '../DATA/GSQPWi'+str(W_i)+'/L'+str(N)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)
else:
    directory = '../DATA/GSrandomWi'+str(W_i)+'/L'+str(N)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)

nomefile = str(PATH_now+'LoschL_'+str(N)+'D_'+str(W)+'seed'+str(seed)+'.dat')
np.savetxt(nomefile, np.real(np.c_[TS, LOSCH]), fmt = '%.9f')

print("Realization completed in {:2f} s".format(time()-ti))
