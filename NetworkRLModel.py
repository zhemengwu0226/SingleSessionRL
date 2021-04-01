import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import random
import copy


def LRate(x, v):
    """
    function for the asymmetric update
    input x will be amplified by v if x >= 0
    otherwise x is unchanged
    """
    if x >= 0:
        return x * v
    else:
        return x

def guessW(W):
    x = np.arange(-5,5,0.1)
    y = 0.5 + 0.5 * erf(x/np.sqrt(2))

    x_out = x[np.where(y < W)]
    return np.max(x_out)

class Participant():
    def __init__(self,
                 alpha = 0.001,  # learning rate
                 sigma = 0.6195, # sigma^-1
                 c=0,            # diffuse parameter
                 v=6.0,          # asymmetry amplifier
                 W_L = [0.7, 0.001, 0.001],    # initial link to E, [port, S_R, S_L]
                 W_R = [0.01, 0.001, 0.001],   # initial link to I, [port, S_R, S_L]
                 asy = 1.0,   # change input x to [1,as,0] or [1,0,as]?
                 use_multi = True):  # if True, use multiplicative model, otherwise use additive model
        self.alpha = alpha
        self.sigma = sigma
        self.c = c
        self.v = v
        self.W_R = np.array(W_R)
        self.W_R_all = [np.array(W_R)]
        self.W_L = np.array(W_L)
        self.W_L_all = [np.array(W_L)]
        self.trials_alls = [0]   # an array that records idx of trials when updates happens
        self.use_multi = use_multi
        self.asy = asy


        # archriving Ws anr accuracy
        self.W_L_Port = [self.W_L[0]]
        self.W_L_L = [self.W_L[1]]
        self.W_L_R = [self.W_L[2]]
        self.W_R_Port = [self.W_R[0]]
        self.W_R_L = [self.W_R[1]]
        self.W_R_R = [self.W_R[2]]

        Corr_left_0 = self.getProbabilities(x=[1,1,0])[0]
        Corr_right_0 = self.getProbabilities(x=[1,0,1])[1]
        self.Corr_Left = [Corr_left_0]
        self.Corr_Right = [Corr_right_0]
        self.Corr_sum = [0.5*(Corr_left_0 + Corr_right_0)]

    def internalQ(self, x):
        # calculate (W_L - W_R) . x, where x = [1,1,0] for left or [1,0,1] for right sound
        # if out > 0, more likely to chose LEFT
        Q = np.dot(np.array(x), self.W_L - self.W_R)
        return Q

    def getProbabilities(self, x=[1,1,0]):
        Q = self.internalQ(x=x)
        P_L = 0.5 + 0.5 * erf(Q / np.sqrt(2))
        return [P_L, 1-P_L]

    def makeChoise(self, x=[1,1,0]):
        P_L, P_R = self.getProbabilities(x=x)
        a = random.uniform(0,1)
        if a <= P_L:
            return [1, 0]  # LEFT
        else:
            return [0, 1]  # RIGHT

    def __getAlphas(self):
        if self.use_multi:
            alpha_L = self.alpha * self.W_L
            alpha_R = self.alpha * self.W_R
        else:
            alpha_L = self.alpha * np.ones(3)
            alpha_R = self.alpha * np.ones(3)

        return alpha_L, alpha_R

    def updateWs(self, x=[1,1,0], Response=[1,0], status=False):
        Q = self.internalQ(x=x)
        alpha_L, alpha_R = self.__getAlphas()
        #assert abs(self.sigma * Q) <= 1.0, "sigma*Q outside [-1,1]?"

        if x[2] == 0:  # x = x1, should choose left
            dW_L = alpha_L * LRate(1 - self.sigma * Q, self.v) * Response[0] * np.array(x) + \
                   self.c * (self.W_R - self.W_L)
            dW_R = alpha_R * LRate(-1 - self.sigma * Q, self.v) * Response[1] * np.array(x) + \
                   self.c * (self.W_L - self.W_R)

        if x[1] == 0: # x = x2, should choose right
            dW_L = alpha_L * LRate(-1 + self.sigma * Q, self.v) * Response[0] * np.array(x) + \
                   self.c * (self.W_R - self.W_L)
            dW_R = alpha_R * LRate(1 + self.sigma * Q, self.v) * Response[1] * np.array(x) + \
                   self.c * (self.W_L - self.W_R)

        if status:
            print("Q is {q:.3f}".format(q=Q))
            print("sigma*Q is {q:.3f}".format(q=Q*self.sigma))
            print("dW_L: {a}".format(a=dW_L))
            print("W_L: {a} --> {b}".format(a=self.W_L, b=self.W_L + dW_L))
            print("dW_R: {a}".format(a=dW_R))
            print("W_R: {a} --> {b}".format(a=self.W_R, b=self.W_R + dW_R))

        self.W_L = self.W_L + dW_L
        self.W_R = self.W_R + dW_R

        self.W_R_all.append(self.W_R)
        self.W_L_all.append(self.W_L)
        self.trials_alls.append(self.trials_alls[-1] + 1)

        self.W_L_Port.append(self.W_L[0])
        self.W_L_L.append(self.W_L[1])
        self.W_L_R.append(self.W_L[2])
        self.W_R_Port.append(self.W_R[0])
        self.W_R_L.append(self.W_R[1])
        self.W_R_R.append(self.W_R[2])

        Corr_left = self.getProbabilities(x=[1, 1, 0])[0]
        Corr_right = self.getProbabilities(x=[1, 0, 1])[1]
        self.Corr_Left.append(Corr_left)
        self.Corr_Right.append(Corr_right)
        self.Corr_sum.append(0.5 * (Corr_left + Corr_right))

        if status:
            print("Accurate rate after updates:")
            print("Left: {l:.3f}; Right: {r:.3f}".format(l=Corr_left, r=Corr_right))

    def evolve(self, bin_trials = 80, n_bins = 28):
        for i in range(n_bins):
            x1, x2 = np.array([1,1,0]), np.array([1,0,1])
            Q1, Q2 = self.internalQ(x=x1), self.internalQ(x=x2)
            Response1, Response2 = self.getProbabilities(x=x1), self.getProbabilities(x=x2)
            alpha_L, alpha_R = self.__getAlphas()

            dW_L_1 = 0.5 * alpha_L * LRate(1 - self.sigma * Q1, self.v) * Response1[0] * np.array(x1)
            # chances for x=x1 is 50%, and P(L|x=x1) is given by Response1[0]
            # under such condition, dW_L will be changed by
            # dW_L = alpha_L * LRate(1 - self.sigma * Q1, self.v) * np.array(x1)
            dW_L_2 = 0.5 * alpha_L * LRate(-1 + self.sigma * Q2, self.v) * Response2[0] * np.array(x2)
            dW_L_diff = self.c * (self.W_R - self.W_L)
            dW_L = dW_L_1 + dW_L_2 + dW_L_diff

            dW_R_1 = 0.5 * alpha_R * LRate(-1 - self.sigma * Q1, self.v) * Response1[1] * np.array(x1)
            dW_R_2 = 0.5 * alpha_R * LRate(1 + self.sigma * Q2, self.v) * Response2[1] * np.array(x2)
            dW_R_diff = self.c * (self.W_L - self.W_R)
            dW_R = dW_R_1 + dW_R_2 + dW_R_diff

            self.trials_alls.append(self.trials_alls[-1] + bin_trials)

            self.W_L = self.W_L + dW_L * bin_trials
            self.W_L_all.append(self.W_L)
            self.W_L_Port.append(self.W_L[0])
            self.W_L_L.append(self.W_L[1])
            self.W_L_R.append(self.W_L[2])

            self.W_R = self.W_R + dW_R * bin_trials
            self.W_R_all.append(self.W_R)
            self.W_R_Port.append(self.W_R[0])
            self.W_R_L.append(self.W_R[1])
            self.W_R_R.append(self.W_R[2])

            Corr_left = self.getProbabilities(x=[1, 1, 0])[0]
            Corr_right = self.getProbabilities(x=[1, 0, 1])[1]
            self.Corr_Left.append(Corr_left)
            self.Corr_Right.append(Corr_right)
            self.Corr_sum.append(0.5 * (Corr_left + Corr_right))

class ExpData():
    def __init__(self, RatID=11):
        #data_path = "/Users/haoyufan/Sunny/SingleContextSequence/Result"
        data_path = "/media/zhemengwu/Gigantic Data/SingleContextSequence/Result/"
        data_file = os.path.join(data_path, "SingleContext_Rat{n}.csv".format(n=RatID))
        self.df = pd.read_csv(data_file)
        self.dates = self.df["Date"].unique()

    def getRLCorr(self, n_bins=1):
        R_corr, L_corr = np.array([]), np.array([])
        for date in self.dates:
            df_day = self.df.loc[self.df["Date"] == date]

            df_L = df_day.loc[df_day["Sequence"] == "A"]
            L_corr_day = df_L["Correct"].to_numpy()
            L_corr_day = L_corr_day[L_corr_day != -99]
            L_corr_day = np.array_split(L_corr_day, n_bins)
            L_corr = np.append(L_corr, [np.mean(item) for item in L_corr_day])

            df_R = df_day.loc[df_day["Sequence"] == "B"]
            R_corr_day = df_R["Correct"].to_numpy()
            R_corr_day = R_corr_day[R_corr_day != -99]
            R_corr_day = np.array_split(R_corr_day, n_bins)
            R_corr = np.append(R_corr, [np.mean(item) for item in R_corr_day])

        return L_corr, R_corr

    def getSumCorr(self, n_bins=2):
        Sum_Corr = np.array([])
        for date in self.dates:
            df_day = self.df.loc[self.df["Date"] == date]

            corr_day = df_day["Correct"].to_numpy()
            corr_day = np.array_split(corr_day, n_bins)
            Sum_Corr = np.append(Sum_Corr, [np.mean(item) for item in corr_day])

        return Sum_Corr


ratID = 16
data = ExpData(RatID=ratID)
L_Corr_exp, R_Corr_exp = data.getRLCorr(n_bins=1)

"""alpha_all = np.arange(6) * 0.002 + 0.001
sigma_all = np.arange(0.50,1.01, 0.05)
v_all = np.arange(1.0,3.1,0.333)
dW_0_all = np.arange(1.00, 2.01, 0.1) # this determine the initial Q and Corr rate
dW_portion_all = np.arange(0, 0.31, 0.1)  # how dW_0 should be arranged between W_LPort and W_LL
W_Rport_all = np.arange(0.001, 10.1, 0.5) # initial W between Port and Right, since dW_0 is "relative"
W_other_all = np.arange(0.01, 0.51, 0.05)"""

alpha_all = np.arange(5)*0.001 + 0.002
sigma_all = np.arange(7)*0.01 + 1.03
v_all = np.arange(5)*0.333 + 2.333
dW_0_all = np.arange(7)*0.01 + 1.16
dW_portion_all = np.arange(3)*0.1 + 0
W_Rport_all = np.arange(7)*0.01 + 9.64
W_other_all = np.arange(3)*0.01 + 0.01
step = 1

best_fit = {"SSE": 9999,
            "alpha": 0,
            "sigma":0,
            "v":0,
            "dW_0":0,
            "W_Rport":0,
            "dW_portion":0,
            "W_other":0,
            "W_L":[0,0,0],
            "W_R":[0,0,0]}

for ii, dW_0 in enumerate(dW_0_all):
    for jj, W_Rport in enumerate(W_Rport_all):
        for kk, dW_portion in enumerate(dW_portion_all):
            W_LPort = W_Rport + dW_0 * (1 - dW_portion)
            for ll, W_other in enumerate(W_other_all):
                pair_all = len(dW_0_all) * len(W_Rport_all) * len(dW_portion_all) * len(W_other_all)
                pair_complete = ii*len(W_Rport_all)*len(dW_portion_all)*len(W_other_all) + \
                                jj*len(dW_portion_all)*len(W_other_all) + \
                                kk*len(W_other_all) + ll
                print("{a:.2f}% complete...".format(a=pair_complete/pair_all*100))
                # ==================================================

                W_LL = dW_0 * dW_portion + W_other
                #print("W_L = [{LP:.2f}, {LL:.2f}, xxx], W_R = [{RP:.2f}, xxx, xxx]".
                #      format(LP = W_LPort, LL = W_LL, RP = W_Rport))
                W_L = [W_LPort, W_LL, W_other]
                W_R = [W_Rport, W_other, W_other]
                for alpha in alpha_all:
                    for sigma in sigma_all:
                        for v in v_all:
                            #print("click")
                            rat = Participant(alpha=alpha, c=0, v=v, sigma=sigma, W_L=W_L, W_R=W_R, use_multi=True)
                            try:
                                rat.evolve(bin_trials=step, n_bins=int(28*80/step))
                            except:
                                continue
                            if np.isnan(rat.Corr_Left).any() or np.isnan(rat.Corr_Right).any():
                                continue

                            L_Corr = rat.Corr_Left[0:-1]
                            L_Corr = np.array_split(L_Corr, 28)
                            L_Corr = np.array([np.mean(item) for item in L_Corr])

                            R_Corr = rat.Corr_Right[0:-1]
                            R_Corr = np.array_split(R_Corr, 28)
                            R_Corr = np.array([np.mean(item) for item in R_Corr])

                            res_L = np.sum(np.abs(L_Corr - L_Corr_exp))
                            res_R = np.sum(np.abs(R_Corr - R_Corr_exp))

                            res = np.abs(L_Corr - L_Corr_exp) + np.abs(R_Corr - R_Corr_exp)
                            SSE = np.sum(res)


                            if SSE < best_fit["SSE"]:
                                old_best = copy.deepcopy(best_fit)
                                best_fit = {"SSE": SSE,
                                            "alpha": alpha,
                                            "sigma": sigma,
                                            "v": v,
                                            "dW_0": dW_0,
                                            "W_Rport": W_Rport,
                                            "dW_portion": dW_portion,
                                            "W_other": W_other,
                                            "W_L": W_L,
                                            "W_R": W_R}

                                print("\n"+"="*30)

                                for key in best_fit.keys():
                                    try:
                                        message = "{key}: {v1:.3f} --> {v2:.3f}".format(key=key,
                                                                                    v1=old_best[key],
                                                                                    v2=best_fit[key])
                                    except:
                                        message1 = "[{v0:.3f}, {v1:.3f}, {v2:.3f}]".format(v0=old_best[key][0],
                                                                                           v1=old_best[key][1],
                                                                                           v2=old_best[key][2])
                                        message2 = "[{v0:.3f}, {v1:.3f}, {v2:.3f}]".format(v0=best_fit[key][0],
                                                                                           v1=best_fit[key][1],
                                                                                           v2=best_fit[key][2])
                                        message = key + ":" + message1 + "-->" + message2
                                    print(message)

                                print("=" * 30)


print("Best Fitted Results")
for key in best_fit.keys():
    try:
        message = "{key}: {v:.3f}".format(key=key, v=best_fit[key])
    except:
        message1 = "[{v0:.3f}, {v1:.3f}, {v2:.3f}]".format(v0=best_fit[key][0],
                                                           v1=best_fit[key][1],
                                                           v2=best_fit[key][2])
        message = key + ":" + message1
    print(message)


rat = Participant(alpha=best_fit["alpha"], c=0, v=best_fit["v"], sigma=best_fit["sigma"],
                  W_L=best_fit["W_L"], W_R=best_fit["W_R"], use_multi=True)

rat.evolve(bin_trials=1, n_bins=28*80)
x = np.arange(28)*80 + 40
plt.scatter(x, L_Corr_exp, color="r")
plt.scatter(x, R_Corr_exp, color="b")
plt.plot(rat.Corr_Left, color="r")
plt.plot(rat.Corr_Right, color="b")
plt.xlabel("SSE = {sse:.3f}".format(sse=best_fit["SSE"]))
plt.show()
