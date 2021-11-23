# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:16:59 2021

@author: Marta Bosnjak
"""
import numpy as np
import math
import sys
import matplotlib.pyplot as plt


class Problem:
        def __init__(self, k,C,g_min,g_max,r_min):
            self.k = k
            self.C=C
            self.g_max=g_max
            self.g_min=g_min
            self.r_min=r_min
        
        def switch_times(self):
            """
            switch_times makes a list of seconds at which the phase changes.
            Note that this function does not always produce valid solutions as
            it does not take into consideration that if a cycle ends with
            green, the next cycle can not start with green

            :returns: 
                - a list of int values that corespond to phase change moments, first element represents the begining time of first green phase
            """ 
            i=0
            times=[]
            pr=np.random.rand()
            if pr>0.5:
                times.append(0)
                i+=1
            while(1):
                if i%2==0:
                    if pr+self.r_min>=self.k*self.C:
                        return times
                    pr=np.random.randint(pr+self.r_min,self.k*self.C)
                else:
                    if pr+self.g_min>=self.k*self.C:
                        return times
                    pr=np.random.randint(pr+self.g_min,pr+self.g_max+1)
                
                if pr>=self.k*self.C:
                    return times
                times.append(pr)
                i+=1
            
        def generate_from_switch_times(self,times):
            """
            generate_from_switch_times makes an array of indicators P where
            P[i] = 0 or P[i] = 1 (0 represents red and 1 represents green) at
            each second of the interval that we are looking to optimize
            
            :parameters:
             - times - a list of int values that corespond to phase change moments, first element represents the begining time of first green phase
            :returns: 
                - np.array P of type int
            """ 
            P=np.zeros(self.k*self.C,dtype=int)
            if len(times)==0:
                return P
            if len(times)==1:
                for i in range(times[0],self.k*self.C):
                    P[i]=1
                return P
            j=1
            while(1):
                if len(times)<j+1:
                    return P
                elif len(times)==j:
                    for i in range(times[j-1],self.k*self.C):
                        P[i]=1
                    return P
                for i in range(times[j-1],times[j]):
                    P[i]=1
                j+=2
            
        
        def valid_test(self,P):
            """
            valid_test tests whether all constraints have been satisfied
            
            :parameters:
             - P - array of indicators P where P[i] = 0 or P[i] = 1 (0 represents red and 1 represents green) at each second of the interval that we are looking to optimize
            :returns: 
                - 1 if all constraints have been satisfied, 0 otherwise
            """ 
            greens = 0
            reds = 0

            if P[0] == 1:
                greens += 1
            else:
                reds += 1
            for i in range(1, self.k * self.C):
                if P[i] == 1:
                    greens += 1
                    if P[i - 1] == 0:
                        if reds < self.r_min:
                            return 0
                        else:
                            reds = 0
                    else:
                        if i % self.C == 0:
                            return 0
                        if greens > self.g_max:
                            return 0
                else:
                    reds += 1
                    if P[i - 1] == 1:
                        if greens < self.g_min:
                            return 0
                        else:
                            greens = 0
            if greens>self.g_max or reds<self.r_min or (P[-1]==1 and greens<self.g_min):
                return 0
            else:
                return 1

        def generate_best_valid_neighbour(self,A,P):
            """
            generate_best_valid_neighbour checks if any of neighbouring 
            solutions of P are valid and selects the best valid one if there
            are multiple
            
            :parameters:
             - A - a list of int values where A[i] represents the number of cars arriving at the second i at a particular phase
             - P - array of indicators P where P[i] = 0 or P[i] = 1 (0 represents red and 1 represents green) at each second of the interval that we are looking to optimize
            :returns: 
                - P_best - of type int if there is at least one valid neighbouring solution, or an empty list if there are none
            """ 
            P_best=np.copy(P)
            crit_best=0
            for i in range(len(P)):
                P_t=np.copy(P)
                if P[i]==1:
                    P_t[i]=0
                else:
                    P_t[i]=1
                if self.criteria(A,P_t)>=crit_best and self.valid_test(P_t):
                    P_best=np.copy(P_t)
                    crit_best=self.criteria(A,P_best)   
            if np.array_equal(P,P_best):
                return []
            else:  
                return P_best
        def generate_best_new(self,A,P):
            """
            generate_best_new generates len(P) new random solutions, and 
            selects the best one
            
            :parameters:
             - A - a list of int values where A[i] represents the number of cars arriving at the second i at a particular phase
             - P - array of indicators P where P[i] = 0 or P[i] = 1 (0 represents red and 1 represents green) at each second of the interval that we are looking to optimize
            :returns: 
                - np.array P_best of type int
            """ 
            P_best=np.copy(P)
            crit_best=0
            for i in range(len(P)):
                P_t=self.generate_from_switch_times(self.switch_times())
                if self.criteria(A,P_t)>=crit_best and self.valid_test(P_t):
                    P_best=np.copy(P_t)
                    crit_best=self.criteria(A,P_best)        
            return P_best
                
        def criteria(self, A , P):
            """
            criteria calculates the sum we are looking to optimize
            
            :parameters:
             - A - a list of int values where A[i] represents the number of cars arriving at the second i at a particular phase
             - P - array of indicators P where P[i] = 0 or P[i] = 1 (0 represents red and 1 represents green) at each second of the interval that we are looking to optimize
            :returns: 
                 - f - sum of element-wise multiplication result of A and P
        
            """
            return np.sum(np.multiply(A, P))
        
        def simulated_annealing(self,A,T,it):
            """
            simulated_annealing performs the search for the optimal
            solution of the problem. Depending on the temperature at the given
            iteration the algorithm decides whether to stay in the same state
            or to switch to the generated worse state. If the generated state
            is better, algorithm always makes the transition.
            
            
            :parameters:
             - A - a list of int values where A[i] represents the number of cars arriving at the second i at a particular phase
            
             - T - array of temperatures 
            
             - it - number of iterations per one temperature value
            
            :returns: 
                - f - criteria value for each iteration
                - P_max - best solution found
                - crit_max - criteria value for the best solution
        
            """
            
            pr_v=[]
            f = np.zeros(len(T) * it) 
            P = self.generate_from_switch_times(self.switch_times())
            P_max=self.generate_from_switch_times(self.switch_times())
            crit_max=self.criteria(A, P_max) if self.valid_test(P_max) else 0
            f[0] = 0
            j=0
            while j < len(T)*it-1:
                j += 1
                P_new=self.generate_best_valid_neighbour(A, P)
                if len(P_new)==0:
                    P_new=self.generate_best_new(A,P)
                
                f_new=self.criteria(A,P_new)
                delta = f_new-f[j-1]
                if delta >= 0:
                    pr_v.append(0)
                    P = np.copy(P_new)
                    if self.criteria(A,P_new) > crit_max and self.valid_test(P_new):
                        crit_max = self.criteria(A,P_new)
                        P_max = np.copy(P_new) 
                            
                else:
                
                    pr = math.exp(delta / (T[int(j/it)] + sys.float_info.epsilon))
                    pr_v.append(pr)
                    random_pr = np.random.rand()
                    if random_pr <= pr:
                        P = np.copy(P_new)
                f[j]=self.criteria(A,P)
                if j % 200 == 0:
                    P= self.generate_from_switch_times(self.switch_times())
                
            fig2, ax2 = plt.subplots()
            ax2.plot(pr_v)
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('pr')

            return f, P_max, crit_max

        def brute_force(self, A):
            P=[None]*self.C*self.k
            solution,max_crit=self.brute_force_recursive(A, P,P,0, 0)
            return solution, max_crit
        
        def brute_force_recursive(self,A,P,solution,max_crit,i):
            if i == self.k * self.C:
                crit_val = self.criteria(A, P)
                if crit_val > max_crit and self.valid_test(P):
                    solution = P[:]
                    max_crit = crit_val
                return solution, max_crit

            P[i] = 0
            solution, max_crit = self.brute_force_recursive(A,P, solution, max_crit, i+1)

            P[i] = 1
            solution, max_crit = self.brute_force_recursive(A,P, solution, max_crit, i+1)

            return solution, max_crit
        
            
k = 3
C = 10

g_min = 5
g_max = 7
r_min = 4
p=Problem(k,C,g_min,g_max,r_min)
A = [1, 1, 3, 4, 3, 3, 3, 1, 2, 1, 1, 1, 1, 1, 3, 5, 3, 10, 9, 9, 20, 25, 30, 0, 0, 0, 0, 0, 0, 0]


# k = 2
# C = 10
#
# g_min =3
# g_max = 7
# r_min = 2
# p=Problem(k,C,g_min,g_max,r_min)
# A=[0,0,3,4,3,3,3,0,2,1,0,0,0,0,0,3,4,3,0,0]


T=np.linspace(3,0,1000)
it=10
crit_max_final=0
for i in range(10):
    f,P_max,crit_max=p.simulated_annealing(A, T, it)
    if crit_max>crit_max_final and p.valid_test(P_max):
        P_max_final=P_max
        crit_max_final=crit_max

print(P_max_final)
print(crit_max_final)

