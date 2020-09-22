import quimb as qu
from quspin.operators import hamiltonian
from quspin.tools.measurements import ent_entropy
from quspin.basis import spin_basis_1d
import numpy as np
import params
import h_functions as hf
from scipy.special import comb
from time import time
import sys,os
import math

### RUN 'solve' METHOD FOR LONG TIMES ###
ti = time() # start timing function for each realization

LOCAL = os.path.abspath('.')
PATH_now = LOCAL

N = params.L
W = params.D
J = 1.0
dis_flag = params.Dis_gen
seed = int(100000*np.random.random())
int_flag = params.Int_flag
t_tab = np.logspace(-1, 3, 400)
if int_flag==0:
    J_tab=(J,J,0)
else:
    J_tab=(J,J,J)

P = qu.zspin_projector(N, sz=0)

if dis_flag==1:
    H_full = qu.ham_mbl(N, W, J_tab, cyclic=False, dh_dist='qp', beta=0.721, seed=seed, sparse=True).real
else:
    H_full = qu.ham_mbl(N, W, J_tab, cyclic=False, seed=seed, sparse=True).real

H = P.T @ H_full @ P
basis = spin_basis_1d(N,Nup=N//2,pauli=False)
sublat_list = [[(-1.0)**i/N,i] for i in range(0,N)]
imbalance_list = [['z', sublat_list]]
no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}

I = hamiltonian(imbalance_list, [], dtype=np.float64, basis=basis, **no_checks)

psi_0 = P.T @ qu.neel_state(N)
subsys = range(N//2) # define subsystem

base=hf.Base_states(N,N//2)
ind_n=np.zeros((len(base), 2))
for i in range(len(base)):
    somma=0
    for j in range(N//2):
        somma+=int(base[i][j])
    ind_n[i]=[somma,i]
StateList=[]
for i in range(N//2+1):
    StateList.append(np.where(ind_n[:,0]==i)[0])

compute = {
    'time': lambda t, p: t,
    'losch': lambda t, p: np.square(np.absolute(qu.fidelity(psi_0, p))),
    'imb': lambda t,p: np.real(I.expt_value(p)),
    'entropy': lambda t, p: ent_entropy(p, basis, chain_subsys=subsys)['Sent_A'],
    'num_ent': lambda t, p: hf.Num_ent(N, p, StateList)
}
evo = qu.Evolution(psi_0, H, compute=compute, method='solve')


for pt in evo.at_times(t_tab):
    ts=evo.results['time']
Sent=N//2*(np.array(evo.results['entropy'])/np.log2(math.e))
P_N=np.array(evo.results['num_ent'])/np.log2(math.e)
Losch=np.array(-np.log(evo.results['losch']))/N
Imb=2*np.array(evo.results['imb'])


if dis_flag == 1:
    directory = '../DATA/NeelQPLongJz'+str(int_flag)+'/L'+str(N)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)
else:
    directory = '../DATA/NeelrandomLongJz'+str(int_flag)+'/L'+str(N)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)

nomefile = str(PATH_now+"L{}W{:.1f}Seed{}.dat".format(N,W, seed))
np.savetxt(nomefile, np.c_[t_tab, Losch, Imb, Sent, P_N], fmt = '%.9f')

print("Realization completed in {:2f} s".format(time()-ti))
