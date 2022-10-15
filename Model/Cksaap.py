import pandas as pd
from collections import Counter
import numpy as np
def Cksaap(filename):
    K=7
    data = pd.read_csv(filename, header=None)
    PepN=np.array(data[1])
    PepN=np.reshape(PepN, (-1, 1))              #(650,)变为（650,1）
    a = PepN[0].tolist()
    ll=len(a[0])
    sap=list()
    AM = 'ACDEFGHIKLMNPQRSTVWYUX';
    for i in range(0, 22):
        for j in range(0, 22):
            sap.append(AM[i]+AM[j]);

    (m, n) = PepN.shape
    saap = np.zeros((m, 484, K))
    for k in range(0,K):
        for i in range(0,m):
            pep = list()
            a=PepN[i];
            b=a[0]
            for j in range(0,len(b)):
                if j+k+1<len(b):
                    pep.append(b[j]+b[j+k+1]);
                elif j+k+1>=len(b):
                    pep.append(b[j]+'X');
            result = Counter(pep)
            ss =list(result.keys())
            val = list(result.values())
            for kk in range(0,len(ss)):
                 mk = sap.index(ss[kk]);
                 saap[i,mk,k] = val[kk];
    allcksaap = np.concatenate((saap[:,:,0],saap[:,:,1],saap[:,:,2],saap[:,:,3],saap[:,:,4],saap[:,:,5],saap[:,:,6]),axis=1)
    return allcksaap
#
import numpy as np
Gluname = r"./GluNALL16_new.csv"
cksaap=Cksaap(Gluname)
np.save('Cksaap.npy',cksaap)

