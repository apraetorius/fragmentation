import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate


#=======================================================
def eq(par,initial_cond,start_t,end_t,incr):
    #-time-grid-----------------------------------
    t  = np.linspace(start_t, end_t,incr)
    #differential-eq-system----------------------
    def funct(y,t):
        P0=y[0]
        P1=y[1]
        P2=y[2]
        P3=y[3]
        k1,k2,k3=par
        # the model equations 
        f0 = -k1*P0-k2*P0-k3*P0
        f1 = +k1*P0
        f2 = +k2*P0
        f3 = +k3*P0
        return [f0, f1, f2, f3]
    #integrate------------------------------------
    ds = integrate.odeint(funct,initial_cond,t)
    return (ds[:,0],ds[:,1],ds[:,2],ds[:,3],t)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import fmin
#=====================================================

#=====================================================

#1.Get Data
#====================================================
Td=np.array([0,1000,2000]).astype(int)#time
M1=np.array([0,0.32,0.49])#mass p1
M2=np.array([0,0.01,0.25])#mass p2
M3=np.array([0,0.08,0.56])#mass p3
#====================================================

#2.Set up Info for Model System
#===================================================
# model parameters
#----------------------------------------------------
k1 = 0.00001       # k1
k2 = 0.000002  # k2
k3 = 0.000003  # k3

rates=(k1,k2,k3)

# model initial conditions
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
print(k1,k2,k3)
#=======================================================

#5.Generate Solution to System
#=======================================================
#F0,F1,F2,F3,T=eq(newrates,y0,start_time,end_time,intervals)
#my_list = [0, 50, 99]
#Zm=F1[my_list]
#Tm=T[my_list]
#======================================================
#print(Zm)
