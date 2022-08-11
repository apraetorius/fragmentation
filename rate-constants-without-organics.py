"""
Created in January

@author: AntoniaPraetorius

written for Python 3
"""
#Code  to derive fragmentation rate constants (k_frag and k_frag-all) from 
#fitting experimental fragmentation data (i.e. released mass of different size
#classes of micro- and nanoplastics upon UV irradiation of microplastics) from
#modified NanoRelease protocol

#this version disregards dissolved organics (for version accounting for loss 
#dissolved organics, see script rate-constants-with-organics.py in same 
#repository)


#import relevant modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import fmin

#define number of size classes (bins) and model equations (assuming first order  kinetics)
#=======================================================
def eq(par,initial_cond,start_t,end_t,incr):
    #-time-grid-----------------------------------
    t  = np.linspace(start_t, end_t,incr)
    #differential-eq-system----------------------
    def funct(y,t):
        P0=y[0] #parent size class
        P1=y[1] #first fragment size class
        P2=y[2] #second fragment size class
        P3=y[3] #third fragment size class
        k1,k2,k3=par
        # the model equations 
        f0 = -k1*P0-k2*P0-k3*P0 #first order fragmentation from parent size class into the three smaller size classes with rate constants k1, k2 and k3
        f1 = +k1*P0 #formation of fragments in first fragment size class
        f2 = +k2*P0 #formation of fragments in second fragment size class
        f3 = +k3*P0 #formation of fragments in third fragment size class
        return [f0, f1, f2, f3]
    #integrate------------------------------------
    ds = integrate.odeint(funct,initial_cond,t)
    return (ds[:,0],ds[:,1],ds[:,2],ds[:,3],t)



#=====================================================

#1.Get Data (experimental data from fragmentation experiment), SI Table 7
#====================================================
Td=np.array([0,1000,2000]).astype(int) #experimental timesteps (in hours)
M1=np.array([0.05,0.32,0.49])#mass in first fragment size class (p1)
M2=np.array([0.02,0.01,0.25])#mass in second fragment size class (p2)
M3=np.array([0.14,0.08,0.56])#mass in third fragment size class (p3)
#====================================================

#2.Set up information for model system
#===================================================
# model parameters
#----------------------------------------------------
k1 = 0.00001   # k1 
k2 = 0.000002  # k2
k3 = 0.000003  # k3

rates=(k1,k2,k3)

# model initial conditions. 
#NOTE: here it is assumed that initial mass of fragments is 0, despite some 
#traces measured at time t=0. To test effect of including initial 
#concentration, model can be re-run with data provided above for M1-M3.
#---------------------------------------------------
P0_0 = 500.               # initial mass p0
P1_0 = 0                  # initial mass p1
P2_0 = 0                  # initial mass p2
P3_0 = 0                  # initial mass p3
y0= [P0_0, P1_0, P2_0,P3_0]      # initial condition vector

# model steps
#---------------------------------------------------
start_time=0.0
end_time=2000.0
intervals=100
mt=np.linspace(start_time,end_time,intervals)

# model index to compare to data
#----------------------------------------------------
findindex=lambda x:np.where(mt>=x)[0][0]
#TO FIX NOW HARDWIRED
mindex=map(findindex,Td)
print(list(mindex))
#=======================================================



#3.Score Fit of System
#=========================================================
def score(parms):
    #a.Get Solution to system
    F0,F1,F2,F3,T=eq(parms,y0,start_time,end_time,intervals)
    #b.Pick of Model Points to Compare
    my_list = [0, 50, 99]
    Zm=F1[my_list]
    Zm2=F2[my_list]
    Zm3=F3[my_list]
#c.Score Difference between model and data points
    ss=lambda data,model:((data-model)**2).sum()
    return ss(M1,Zm)+ss(M2,Zm2)+ss(M3,Zm3)
#========================================================


#4.Optimize Fit
#=======================================================
fit_score=score(rates)
answ=fmin(score,(rates),full_output=1,maxiter=10000)
bestrates=answ[0]
bestscore=answ[1]
k1,k2,k3=answ[0]
newrates=(k1,k2,k3)
print(k1,k2,k3) #print fragmentation rate constants
#=======================================================

#5.Generate Solution to System
#=======================================================
#F0,F1,F2,F3,T=eq(newrates,y0,start_time,end_time,intervals)
#my_list = [0, 50, 99]
#Zm=F1[my_list]
#Tm=T[my_list]
#======================================================
#print(Zm)
