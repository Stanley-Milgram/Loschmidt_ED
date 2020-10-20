import numpy as np
import glob
from tqdm import tqdm
import os

LOCAL = os.path.abspath('.')
PATH_now = LOCAL
PATH_now

Dis_tab=['random', 'QP']
W_tab=[2.0, 7.0, 8.0, 15.0]
int_tab=[0,1]
for dis in Dis_tab:
    print("Averaging the "+dis+" data ....")
    for inter in int_tab:
        print("Averaging for J_z="+str(inter)+" ....")
        directory = 'DataMean/LoschMBL/Neel'+dis+'LongJz'+str(inter)+'/'
        PATH_now = LOCAL+os.sep+directory+os.sep
        if not os.path.exists(PATH_now):
            os.makedirs(PATH_now)
        for w in W_tab:
            print("Averaging different realizations for W="+str(w)+" ....")
            files = sorted(glob.glob("../DATA/Neel"+dis+"LongJz"+str(inter)+"/L14/D"+str(w)+"/*Seed*"))
            numreal=len(files)

            LOSCH=[]
            ENT=[]
            IMB=[]
            NUM_ENT=[]

            for i in tqdm(range(numreal)):
                dat=np.loadtxt(files[i])
                ts=dat[:,0]
                losch=dat[:,1]
                imb=2*dat[:,2]
                ent=dat[:,3]
                num_ent=dat[:,4]
                LOSCH.append(losch)
                ENT.append(ent)
                IMB.append(imb)
                NUM_ENT.append(num_ent)

            Losch_mean=np.mean(LOSCH, axis=0)
            Ent_mean=np.mean(ENT, axis=0)
            Imb_mean=np.mean(IMB, axis=0)
            Num_ent=np.mean(NUM_ENT, axis=0)
            
            np.savetxt(PATH_now+"LoschL14D{:.1f}.dat".format(w), np.c_[ts, Losch_mean,Imb_mean, Ent_mean, Num_ent])
