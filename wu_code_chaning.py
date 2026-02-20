### YUE
import numpy as np
import pandas as pd
import rpy2.robjects
from rpy2.robjects import r, pandas2ri, Formula
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
pandas2ri.activate()
base = rpackages.importr('base')
utils = rpackages.importr('utils')
# ERROR: dependencies ‘survC1’, ‘doBy’, ‘statmod’, ‘shiny’, ‘rootSolve’ are not available for package ‘frailtypack’
installed_list=[]
for xx in rpackages.InstalledPackages():
    installed_list.append(xx[0])

# order matters
for pkg in ["survC1", "doBy", "statmod", "shiny", "rootSolve", "frailtypack"]:
    if pkg not in installed_list:
        if pkg=="frailtypack":
            utils.install_packages('https://cran.r-project.org/src/contrib/Archive/frailtypack/frailtypack_3.6.0.tar.gz', type='source')
        else:
            utils.install_packages(pkg)
# utils.install_packages('survC1')
# utils.install_packages('doBy')
# utils.install_packages('statmod')
# utils.install_packages('shiny')
# utils.install_packages('rootSolve')
# utils.install_packages('https://cran.r-project.org/src/contrib/Archive/frailtypack/frailtypack_3.6.0.tar.gz', type='source')
frailtypack = rpackages.importr('frailtypack')
pd.DataFrame.iteritems = pd.DataFrame.items
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import sys
import shutil
import csv
##############################################################################
# Usage:
# 1. python mono_yue.py                 # this runs default data file
# 2. python mono_yue.py DataFileName    # this runs Input data file
#
# ***Input: A data file name.
#         Data file format:  Ti Li Ri
#         corresponding to left truncation time and two endpoints of censoring
#         interval, e.g. that
#         0 1 2
#         1 2 99999
#         2 3 3
#         represents there are three obs: one is censored b/w 1 and 2 with
#         left truncation time at 0; the second is right censored at 2 with
#         left truncation time at 1; the third is exactly observed at 3.
##############################################################################
def c(a, b):
    return np.concatenate((a, b))
def rep(val, n):
    return np.ones(n,)*val
'''
/* make a Greatest Convex Minorant from
   interval hazards from output of N-C-P.

   March 14, 96
*/
'''
def gcm(q, nh, n):
    nh[0] = q[0] = 0
    for i in range(2, n+1):
        nh[i] += nh[i-1]
    i = 0
    while i < n-1:
        min_slope = (nh[i+1]-nh[i]) / (q[i+1]-q[i])
        last = i + 1
        for j in range(i+2, n+1):
            slope = (nh[j] - nh[i]) / (q[j] - q[i])
            if slope < min_slope:
                min_slope = slope
                last = j

        for k in range(i+1, last):
            nh[k] = nh[i] + min_slope * (q[k] - q[i])

        i = last
    for i in range(n, 1, -1):
        nh[i] -= nh[i-1]
    return nh

def plwp(INfname="default", data=None, df=None, factor=0, knots=4,kappas=1000,quiet=False):
    # initialize the
    # since everything in the code starts from index 1
    # here we need to init LT, LC, RC as [0]
    LT, LC, RC, Cov, Cens = [0], [0], [0], [0], [0]
    N = 0
    Tolerance=0.0001
    Cov1 = [0]

    ###########################
    # load data from the file #
    ###########################
    if data is None:
        with open(INfname) as f:
            lines = f.readlines()
        for line in lines:
            if len(line)>0:
                # and parse the line and append to LT, LC, RC
                res = line.strip().split()
                LT.append(float(res[0]))
                LC.append(float(res[1]))
                RC.append(float(res[2]))
                Cov.append(float(res[3]))
                #Cov1.append(float(res[4]))
                #Cens.append(float(res[5]))
                Cens.append(float(res[4]))
                N+=1
                # if left right interval value are equal
                # shift the right interval a little bit
                if LC[N]==RC[N]:
                    RC[N] += TB_EPSILON
                if N>=MAX_N_Obs:
                    exit("too large data set!")
    else:
        for line in data:
            LT.append(line[0])
            LC.append(line[1])
            RC.append(line[2])
            N+=1
            # if left right interval value are equal
            # shift the right interval a little bit
            if LC[N]==RC[N]:
                RC[N] += TB_EPSILON
            if N>=MAX_N_Obs:
                exit("too large data set!")

    ####################################################
    # z_k (get sorted distinct values from LT, LC, RC) #
    ####################################################
    # TODO (not sure why they have Y but not used anywhere)
    # distinct z_k
    #print(len(lines), len(LT), )
    lr = np.sort(np.unique(np.array(LT[1:]+LC[1:]+RC[1:])))
    p = np.concatenate((np.array([0]), lr[:-1]))
    q = np.concatenate((np.array([0]), lr[1:]))
    print(p)
    # interval lengths between z_i and z_{i+1}
    dt = q - p
    n = lr.size-1
    if n > MAX_N_Obs * 2:
        exit("too large data set (i.e. bins)!")

    # initialize weights for isotonic regression
    nh = np.ones(n+1) / n
    h = np.ones(n+1) / n
    w = np.array(range(0, n+1))  # 0,1,2.,,,,n

    # compute L0=logL(h0)
    # ds are like intervals Is
    d1 = np.zeros(N+1)
    d2 = np.zeros(N+1)
    L0 = 0
    RC1 = np.where(RC==9999, LC, RC)
    df_dict = {"truncation":LT[1:] , "LC":LC[1:], "RC":RC1[1:], "Cov":Cov[1:], "event": Cens[1:]}
    #df_dict = {"truncation":LT[1:] , "LC":LC[1:], "RC":RC1[1:], "Cov":Cov[1:],"Cov1":Cov1[1:], "event": Cens[1:]}
    
    df = pd.DataFrame(data=df_dict)
    #Cox model
    res=frailtypack.frailtyPenal(Formula('SurvIC(t0=truncation,lower=LC,upper=RC,event)~Cov'),data=df,n_knots=knots,kappa=kappas)
    #res=frailtypack.frailtyPenal(Formula('SurvIC(t0=truncation,lower=LC,upper=RC,event)~Cov+Cov1'),data=df,n_knots=knots,kappa=kappas)
    if factor == 0:
        beta_hat = 0
        #beta_hat1 = 0
    else:
        beta_hat = res.rx2('coef')[0]
        #beta_hat1 = res.rx2('coef')[1]
    #print("#"*5,beta_hat, beta_hat1)
    print("#"*5,beta_hat)
    #conditional_print("beta_hat:%.3f", beta_hat)
    for i in range(1, N+1):
        for j in range(1, n+1):
            if p[j]>=LT[i] and q[j] <= LC[i]:
                #print(nh[j], dt[j], beta_hat, df['Cov'][i-1], factor)
                #d1[i] += nh[j] * dt[j] * np.exp(beta_hat*df['Cov'][i-1]*factor+beta_hat1*df['Cov1'][i-1] * factor)
                d1[i] += nh[j] * dt[j] * np.exp((beta_hat*df['Cov'][i-1]) * factor)
            elif p[j]>=LC[i]:
                break
        j_past = j
        for j in range(j_past, n+1):
            if p[j]>=LC[i] and q[j]<=RC[i]:
                d2[i] += nh[j] * dt[j] * np.exp((beta_hat*df['Cov'][i-1]) * factor)
                #d2[i] += nh[j] * dt[j] * np.exp(beta_hat*df['Cov'][i-1]*factor+beta_hat1*df['Cov1'][i-1] * factor)
                # if d2[i]==0:
                #print("i=",i,d2[i],nh[j],dt[j],beta_hat,df['Cov'][i-1],factor)
            elif p[j] >= RC[i]:
                break
        #print("d2[%d]=%f"%(i, d2[i]))
        # if d2[i] < TB_EPSILON:
        #     if not quiet:
        #         print("d2[%d]=%f"%(i, d2[i]))
        #     d2[i] += TB_EPSILON
        d2[i] = np.exp(-d2[i])
        L0 += np.log(1.0-d2[i])-d1[i]

    ###########################################
    # gradient projection algorithm main loop #
    ###########################################
    iter_ = 0
    while iter_ < TB_MAX_ITER:
        iter_ += 1
        # find the gradient for the loglikelihood
        dL = np.zeros(n+1)

        for i in range(1, N+1):
            for j in range(1, n+1):
                if p[j] >= LT[i] and q[j] <= RC[i]:
                    if p[j] >= LC[i]:
                        #dL[j] += d2[i] / (1-d2[i])
                        dL[j] += d2[i] / (1-d2[i])*np.exp(beta_hat*df['Cov'][i-1]* factor)
                        #dL[j] += d2[i] / (1-d2[i])*np.exp(beta_hat*df['Cov'][i-1]* factor+beta_hat1*df['Cov1'][i-1]* factor)
                        #print("dL[%d]+= %.3f / %.3f -> %.3f"%(j, d2[i], 1-d2[i], dL[j]))
                    else:
                        #dL[j] -= 1
                        dL[j] -= np.exp(beta_hat*df['Cov'][i-1]* factor)
                        #dL[j] -= np.exp(beta_hat*df['Cov'][i-1]* factor+beta_hat1*df['Cov1'][i-1]* factor)
                        #print("dL[%d]-= 1 -> %.3f"%(j, dL[j]))
                else:
                    if p[j]>=RC[i]:
                        break

        dL[1:n+1] *= dt[1:n+1]

        # this is the step 2 in paper before projection
        # lambda^{(i)} + \nabla \log L(lambda^{(i)})
        nh[1:n+1] = h[1:n+1] + dL[1:n+1]
        
        #conditional_print("Iter %05d before gcm| h:%s, dL:%s, nh:%s", iter_, nice(h[1:n+1]),nice(dL[1:n+1]), nice(nh[1:n+1]))

        # Add CVM to get the projection P(h[j]+df[j])
        nh = gcm(w, nh, n)
        #conditional_print("Iter %05d after gcm | nh:%s", iter_, nice(nh[1:n+1]))

        # there might be negative ones!
        nh = np.clip(nh, 0, 9999)
        # direction pi=P(xi+df(xi))-xi
        gd = nh - h

        # /**************Amiju Stepsize *****************/
        dLgd = np.sum(dL[1:n+1] * gd[1:n+1])
        #conditional_print("Iter %05d Amiju init | gd:%s", iter_, nice(gd[1:n+1]))
        #conditional_print("Iter %05d Amiju init | nh:%s", iter_, nice(nh[1:n+1]))
        #conditional_print("Iter %05d Amiju init | dLgd:%.3f", iter_, dLgd)        
        stepsize = 1
        for k in range(1, 100000):
            # compute \log L(lambda^{(i)} + alpha^{(i)} p^{(i)})
            # the result is saved in Li
            Li = 0
            nh[1:n+1]=h[1:n+1] + stepsize * gd[1:n+1]
            #conditional_print("Iter %05d Amiju (%05d) | nh:%s", iter_, k, nice(nh[1:n+1]))
            #conditional_print("Iter %05d Amiju init | gd:%s", iter_, nice(gd[1:n+1]))
            #conditional_print("Iter %05d Amiju init | h:%s", iter_, nice(h[1:n+1]))
            d1 = np.zeros(N+1)
            d2 = np.zeros(N+1)

            for i in range(1, N+1):
                d1[i] = 0
                for j in range(1, n+1):
                    if p[j] >= LT[i] and q[j] <= LC[i]:
                        #d1[i] += nh[j] * dt[j]
                        d1[i] += nh[j] * dt[j] * np.exp(beta_hat*df['Cov'][i-1] * factor)
                        #d1[i] += nh[j] * dt[j] * np.exp(beta_hat*df['Cov'][i-1] * factor+beta_hat1*df['Cov1'][i-1]* factor)
                    elif p[j] >= LC[i]:
                        break
                j_past = j
                for j in range(j_past, n+1):
                    if p[j] >= LC[i] and q[j] <= RC[i]:
                        #d2[i] += nh[j] * dt[j]
                        d2[i] += nh[j] * dt[j] * np.exp(beta_hat*df['Cov'][i-1] * factor)
                        #d2[i] += nh[j] * dt[j] * np.exp(beta_hat*df['Cov'][i-1] * factor+beta_hat1*df['Cov1'][i-1]* factor)
                    elif p[j] >= RC[i]:
                        break

                if d2[i] < TB_EPSILON:
                    if not quiet:
                        print("d2[%d]=%f"%(i, d2[i]))
                    d2[i] += TB_EPSILON

                d2[i] = np.exp(-d2[i])
                Li += np.log(1.0-d2[i])-d1[i]
            #conditional_print("Iter %05d Amiju init | d2:%s", iter_, nice(d2))
            #conditional_print("Iter %05d Amiju init | d1:%s", iter_, nice(d1))
            
            # if the condition in step 3 is satisfied, stop
            #conditional_print("Iter %05d Amiju (%05d) | Li - L0:%.3f| Li:%.3f| L0:%.3f|break_val:%.3f", iter_, k, Li - L0,Li,L0,stepsize * dLgd / 4)
            if Li - L0 >= stepsize * dLgd / 4:
                break
            # otherwise, decrease the stepsize by half
            stepsize /= 2
            #conditional_print("Iter %05d Amiju (%05d) | stepsize:%.3f", iter_, k, stepsize)
            if stepsize < TB_EPSILON:
                break

        # some convergence checking stop criteria
        Delta = Li - L0
        #conditional_print("Iter %05d after Amiju | Delta:%.3f", iter_, Delta)
        if Delta < Tolerance and iter_ >= TB_MIN_ITER:
            break
        else:
            h[1:n+1] = nh[1:n+1]
        L0 = Li
        # print("L0=", L0)

    # print result
    # Sat Oct 19 19:00:54 CDT 1996
    if not quiet:
        print("\n\n************RESULT:(after %d iterations)\n\n" % (iter_))
    OUTfname = ""
    OUTfname = OUTfname + INfname
    OUTfname2 = "IfrRes%f" % (Tolerance)
    OUTfname = OUTfname + OUTfname2

    # combine together the intervals with same hazards . May 22, 96
    min_val=nh[1]
    if not quiet:
        print("%f "%p[1])
    for j in range(2, n+1):
        if nh[j] > min_val:
            if not quiet:
                print("%f %f"%(q[j-1], min_val))
                print("%f"%(p[j]))
            min_val = nh[j]
    if not quiet:
        print("%f %f"%(q[n], min_val))

    ### find unique and start-with-last-zero
    int_d = {}
    for j in range(1, n+1):
        nh_key = np.round(nh[j], decimals=5)
        if nh_key not in int_d:
            int_d[nh_key] = []
        int_d[nh_key].append(p[j])
        int_d[nh_key].append(q[j])

    print("nh", nh)
    print("p", p)
    print("q", q)

    nh_short = [0]
    p_short = [0]
    q_short = [0]
    keys = sorted(int_d.keys())
    for key in keys:
        nh_short.append(key)
        times = sorted(np.unique(int_d[key]))
        if key==0:
            p_short.append(times[-2])
            q_short.append(times[-1])
        else:
            p_short.append(times[0])
            q_short.append(times[-1])
    q_short[-1]=q[-2]
    print("nh_short", nh_short)
    print("p_short", p_short)
    print("q_short", q_short)

    #nh, p, q, nh_short, p_short, q_short
    #nh, p, q, beta_hat, beta_hat1, df
    return nh, p, q, beta_hat, df

def baseline_survival_sim(t1, q, hazard_est):
    I = np.digitize(np.array([t1]), bins=q, right=True)-1
    distinct_time = np.concatenate((np.zeros(1), q))
    if t1 <= q[0]:
        cumhaz1 = 0
        rest=t1 * hazard_est[0]
    elif t1 > q[-1]:
        cumhaz1 = 99
        rest = 0
    else:
        cumhaz1 = np.cumsum(np.diff(distinct_time) * hazard_est)[I]
        rest = (t1-q[I]) * hazard_est[I+1]
    intergation = cumhaz1 + rest
    return np.exp(-intergation)






# Hyper-parameters settings
TB_EPSILON = 0.000001   #1 # 100 #0.1
TB_MAX_ITER = 500
TB_MIN_ITER = 5
MAX_N_Obs = 1500

# Working flow logic
MERGE_DATA = True
DRAW_PLOT = True
np_data_path = None

import os
import sys
import shutil
import csv
import pandas as pd

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        # self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
        self.log = open("%s.txt"%(FILE_STRING), "a", 1)        

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# nh_wei, p_wei, q_wei=plwp(INfname="chaning_men.txt", factor=0, knots=4,kappas=1000,quiet=True)
# nh, p, q = plwp(INfname="chaning.txt", factor=1, knots=4, kappas=1000,quiet=True)
# # nh, p, q = nh_wei, p_wei, q_wei

#nh_wei, p_wei, q_wei, beta_hat, beta_hat1, df = plwp(INfname="dat_new_baseline.txt", factor=0, knots=4, kappas=1000, quiet=True)
#nh, p, q, beta_hat, beta_hat1, df = plwp(INfname="dat_new.txt", factor=1, knots=4, kappas=1000, quiet=True)


import argparse
parser = argparse.ArgumentParser("")
add = parser.add_argument
add("--seed", type=int, default=1007)
add("--path1", '-P1', type=str, default="channing_men10_c.txt")
add("--path2", '-P2', type=str, default="channing_data10_c.txt")
add("--path3", '-P3', type=str, default="channing_men10_c.txt")
add("--path4", '-P4', type=str, default="channing_men10_c.txt")
add("--img_path", '-I', type=str, default="chaning.png")
args=parser.parse_args()

nh_wei, p_wei, q_wei, beta_hat, df = plwp(INfname=args.path4, factor=0, knots=10, kappas=0, quiet=True)
#nh_wei_m, p_wei_m, q_wei_m, beta_hat,beta_hat1, df = plwp(INfname=args.path3, factor=0, knots=4, kappas=1000, quiet=True)
#nh_wei_np, p_wei_np, q_wei_np, beta_hat,beta_hat1, df = plwp(INfname=args.path4, factor=0, knots=4, kappas=1000, quiet=True)
nh, p, q, beta_hat, df = plwp(INfname=args.path2, factor=1, knots=10, kappas=0, quiet=True)
print(nh_wei)
print(nh)

# min_t = np.min([np.max(q), np.max(q_wei)])
# min_x = np.max([np.min(q[1]), np.min(q_wei[1])])
# times = np.linspace(min_x, min_t, 100)#shape=6
#wu_item_list=[]
#wei_item_list=[]

##min_t0 = np.max(q_wei)
# min_x0 = np.min(q_wei[1])
#min_x0 = np.min(p_wei[1])
#times0 = np.linspace(min_x0, min_t0, 100)#shape=6
#print(times0)
#min_t1 = np.max(q)
# min_x1 = np.min(q[1])
#min_x1 = np.min(p[1])
#times1 = np.linspace(min_x1, min_t1, 100)#shape=6
#print(times1)

#wu_item_list01 = []
#wu_item_list10 = []
#wu_item_list11 = []

#for i in range(len(p)-1):
#    tt1 = time.time()
    # n_samples = len(loaded_d["wu_weibull"][trial_i])
#    wu_weibull = nh
#    wei_weibull = nh_wei
    # wei_weibull = loaded_d["wei_weibull"][trial_i]
#    wu_zk = q
#    wei_zk = q_wei
    # wei_zk = loaded_d["wei_zk"][trial_i]
#    wu_item=baseline_survival_sim(t1=p[i+1],q=np.array(wu_zk),hazard_est=np.array(wu_weibull))
#    wei_item=baseline_survival_sim(t1=times0[i],q=np.array(wei_zk),hazard_est=np.array(wei_weibull))
#    wu_item = wu_item.item()
#    wei_item = wei_item.item()
#    wu_item_list.append(wu_item)
#    wei_item_list.append(wei_item)
    
    
#    wu_item_01 = np.exp(np.log(wu_item) * np.exp(beta_hat)) 
#    wu_item_10 = np.exp(np.log(wu_item) * np.exp(beta_hat1)) 
#    wu_item_11 = np.exp(np.log(wu_item) * np.exp(beta_hat + beta_hat1)) 
    
#    wu_item_list01.append(wu_item_01)
#    wu_item_list10.append(wu_item_10)
#    wu_item_list11.append(wu_item_11)

#baseline:00->wu_item (current line)
#01 Male&Yes:wu_item**(exp(beta1*cov1))
#10 Female&No: wu_item**(exp(beta*cov))
#11 Female&Yes:  wu_item**(exp(beta*cov+beta1*cov1))
#S_hat_w_=S_0**exp(beta*cov+beta1*cov1)
nh = nh[1:-1]
p = p[1:-1]
q = q[1:-1]
nh_wei = nh_wei[1:-1]
p_wei = p_wei[1:-1]
q_wei = q_wei[1:-1]

# nh_wei_m = nh_wei_m[1:-1]
# p_wei_m = p_wei_m[1:-1]
# q_wei_m = q_wei_m[1:-1]

# nh_wei_np = nh_wei_np[1:-1]
# p_wei_np = p_wei_np[1:-1]
# q_wei_np = q_wei_np[1:-1]
# compute for S(x) based on the intervals (p, q)
integral = np.cumsum(nh * (q-p))
s = np.exp(-integral)

# Cov1 = np.array(df['Cov1'])
# Cov = np.array(df['Cov'])

# print("CHECK",beta_hat, beta_hat1, np.sum(beta_hat1*Cov1), np.sum(beta_hat*Cov))

# s_m=s**(np.exp(beta_hat1))
# s_np=s**(np.exp(beta_hat))

print(s)
# print(s_m)
# print(s_np)

integral_wei = np.cumsum(nh_wei * (q_wei-p_wei))
s_wei = np.exp(-integral_wei)

# integral_wei_m = np.cumsum(nh_wei_m * (q_wei_m-p_wei_m))
# s_wei_m = np.exp(-integral_wei_m)

# integral_wei_np = np.cumsum(nh_wei_np * (q_wei_np-p_wei_np))
# s_wei_np = np.exp(-integral_wei_np)

linewidth=1
print(s_wei)
# print(s_wei_m)
# print(s_wei_np)

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 3, 1)
plt.plot(q, s, color="red", linestyle="-.", label="S_wu_men")
plt.plot(q_wei, s_wei, color="black", linestyle="-", label="S_wei_men")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Survival")

# plt.subplot(1, 3, 2)
# plt.plot(q, s_m, color="red", linestyle="-.", label="S_wu_men")
# plt.plot(q_wei_m, s_wei_m, color="black", linestyle="-", label="S_wei_men")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Survival")

# plt.subplot(1, 3, 3)
# plt.plot(q, s_np, color="red", linestyle="-.", label="S_wu_np")
# plt.plot(q_wei_np, s_wei_np, color="black", linestyle="-", label="S_wei_np")
#plt.plot(times1, wu_item_list, label="wu", color="red", linestyle="--",linewidth=linewidth)
#plt.plot(times0, wei_item_list, label="wei", color="blue", linestyle="--",linewidth=linewidth)
#plt.plot(times1, wu_item_list01, label="wu01", color="orange", linestyle="--",linewidth=linewidth)
#plt.plot(times1, wu_item_list10, label="wu10", color="green", linestyle="--",linewidth=linewidth)
#plt.plot(times1, wu_item_list11, label="wu11", color="purple", linestyle="--",linewidth=linewidth)
# plt.plot(times, wei_avg, label="wei", color="blue", linestyle=":", linewidth=linewidth)
# plt.plot(times, true_prob, label="true", color="black", linestyle='-', linewidth=linewidth)
# plt.title('shape=%d; bias: %.4f | %.4f  MCSE: %.4f | %.4f avg_n: %.1f IC_rate: %.2f %.2f'%(shape, 
    # np.mean(surv_median_est)-0.5,np.mean(surv_median_wei)-0.5,
    # np.std(surv_median_est,ddof=1),np.std(surv_median_wei,ddof=1), np.mean(sample_size),np.mean(event_rate),np.mean(wei_rate)), fontsize=10)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Survival")
#plt.show()
#plt.savefig("channing10.png")
plt.savefig(args.img_path)
plt.close()
