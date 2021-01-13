import quimb as qu
import numpy as np
import params
from time import time
import sys,os
import math


LOCAL = os.path.abspath('.')
PATH_now = LOCAL

ti = time() # start timing function for each realization

### Global parameters ###
N = params.L
W = params.D
J = 1.0
dis_flag = params.Dis_gen
seed = int(1000000*np.random.random())
int_flag = 1
t_tab = np.logspace(-2, 1.5, 300)

### ETH ---> MBL ###
J_ETH=(J,J,J)

P = qu.zspin_projector(N, sz=0)

H_ETH = P.T @ qu.ham_heis(N, J_ETH, sparse=True, cyclic=False) @ P
Psi_ETH = qu.eigvecsh(H_ETH, k=1, which='SA')

J_evo1=(0.0, 0.0, 0.0)
H_evo1 = P.T @ qu.ham_mbl(N, W, J_evo1, cyclic=False, dh_dist='qp', beta=0.721, seed=seed, sparse=True).real @ P

compute = {
    'time': lambda t, p: t,
    'losch': lambda t, p: qu.fidelity(Psi_ETH, p)
}
evo_ETH = qu.Evolution(Psi_ETH, H_evo1, compute=compute, method='expm')

for t in evo_ETH.at_times(t_tab):
    continue

TS=evo_ETH.results['time']
LOSCH_ETH=np.array(-np.log(evo_ETH.results['losch']))/N


### MBL --> ETH ###
W_i = params.W_i

J_MBL=(0.0, 0.0, 0.0)

H_MBL = P.T @ qu.ham_mbl(N, W_i, J_MBL, cyclic=False, dh_dist='qp', beta=0.721, seed=seed, sparse=True).real @ P
Psi_MBL = qu.eigh(H_MBL, k=1, which='SA')[1]

J_evo2=(J,J,J)

H_evo2 = P.T @ qu.ham_mbl(N, W, J_evo2, cyclic=False, dh_dist='qp', beta=0.721, seed=seed, sparse=True).real @ P

compute = {
    'time': lambda t, p: t,
    'losch': lambda t, p: qu.fidelity(Psi_MBL, p)
}
evo_MBL = qu.Evolution(Psi_MBL, H_evo2, compute=compute, method='expm')

for t in evo_MBL.at_times(t_tab):
    continue

LOSCH_MBL=np.array(-np.log(evo_MBL.results['losch']))/N

if dis_flag == 1:
    directory = '../DATA/TestGSQPWi'+str(W_i)+'/L'+str(N)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)
else:
    directory = '../DATA/TestGSrandom/L'+str(N)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)

nomefile = str(PATH_now+'LoschL_'+str(N)+'D_'+str(W)+'seed'+str(seed)+'.dat')
np.savetxt(nomefile, np.real(np.c_[TS, LOSCH_ETH, LOSCH_MBL]), fmt = '%.9f')

print("Realization completed in {:2f} s".format(time()-ti))
