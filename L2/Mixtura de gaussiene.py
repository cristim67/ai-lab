# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:11:10 2024

@author: Laborator
"""


import numpy as np
from skimage import io,color
import matplotlib.pyplot as plt

plt.close('all')


PI = 3.1415
eps = 0.00000001
L=256

# functie care genereaza valorile unei/unor distributii gaussiene pentru care se dau medie, varianta, probabilitate apriori
def my_gaussian(m,v,prob):  #medie -> m, varianta -> v, probabilitate -> prob
    K=np.size(m)
    prob=prob/np.sum(prob)
    pdf=np.zeros([256,K])
    coef=prob/(np.sqrt(2*PI*v))
    for i in range(0,256):
        pdf[i,:]=coef*np.exp(-(i-m)*(i-m)/(2*v))
    hmixt = np.sum(pdf,axis=1)
    hmixt=hmixt/sum(hmixt)
    return hmixt, pdf


# functia care face descompunerea unei histograme in moduri gaussiene
def my_EM(hist,K):
    p=np.ones(K)/K
    mu=L*np.array(range(1,K+1))/(K+1);
    v= L*np.ones(K)
    while(1):
        hmixt,prb = my_gaussian(mu,v,p);
        scal = np.sum(prb,axis=1)+eps;
        loglik=np.sum(hist*np.log(scal));
        for i in range(0,K):
            pp=hist*prb[:,i]/scal
            p[i]=np.sum(pp)
            mu[i]=np.sum(np.array(range(0,L))*pp)/p[i]
            vr=np.array(range(0,L))-mu[i]
            v[i]=np.sum(vr*vr*pp)/p[i]
        p=p/np.sum(p)

        hmixt,prb = my_gaussian(mu,v,p);
        scal = np.sum(prb,axis=1)+eps;
        nloglik=np.sum(hist*np.log(scal));
        if((nloglik-loglik)<0.0001):
            break
    #gata while
    return mu, v, p


# de aici putem genera mixturile:
# exemplu: 3 gaussiene, prima cu media 100, varianta 400, acopera 50%, a doua cu media 150, varianta 100, acopera 25%, a treia cu media 200, varianta 1600, acopera 25%
medii=np.array([100,150,200])
variante=np.array([400,100,1600])
dispersii=np.sqrt(variante)
probabilitati=np.array([0.5,0.25,0.25])
hmixt, g = my_gaussian(medii, variante, probabilitati)
plt.figure(),plt.plot(g),plt.plot(hmixt),plt.show()

# aici se ia histo care este o histograma de sinteza sau histograma unei imagini
# si se descompune cu EM
m1, v1, p1 = my_EM(hmixt,1)
print(m1)
print(v1)
print(p1)
histo1, g1 = my_gaussian(m1, v1, p1)
plt.figure(),plt.plot(hmixt),plt.plot(histo1),plt.show()
plt.figure(),plt.plot(g1),plt.show()



def Nakagawa(hmixt,medii,dispersii):
    cond=True
    k=len(medii)
    for i in range (0,k-1):
        if medii[i+1]-medii[i]<=4:
            cond=False
        if (0.1>=(dispersii[i+1]/dispersii[i])) or (10<=(dispersii[i+1]/dispersii[i])):
            cond=False
        if np.min(hmixt[medii[i]:medii[i+1]])>0.8*min(hmixt[medii[i]],hmixt[medii[i+1]]):
            cond=False  
        if cond==True:
            print("Sunt separabile:",i,i+1)
        else:
            print("Nu sunt separabile:",i,i+1)
    
Nakagawa(hmixt,medii,dispersii)    
    