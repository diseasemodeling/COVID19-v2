# -*- coding: utf-8 -*-

import pandas as pd
from math import isnan
import numpy as np
"""
global default_values 
default_values = [0.1,0.2,0.3]
# input should from samll values to larger values
# input date should be less than Tmax
# day should strat from 1

def read_policy(path,Tmax):
    df = pd.read_excel(path, header =0)
    
    params = df.values.shape[1]//3
    l = []
    for k in range(params):
        l.append([])
    
    for i in range(df.values.shape[0]):
        for j in range(params):
        # policy 1
            if not isnan(df.values[i][params*j]):
                l[j].append((int(df.values[i][params*j])-1,int(df.values[i][params*j+1]),df.values[i][params*j+2]))
    
    policy = np.zeros((Tmax,params))
    
    for j in range(params):
        i = 0
        if len(l[j]) != 0:
            for i in range(l[j][0][0]):
                policy[i][j] =  default_values[j]
            i += 1
            v = l[j]
            for k in range(len(v)):
                if k == 0:
                    pre_value = default_values[j]
                else:
                    pre_value = v[k -1][2]
                    
                while i < v[k][0]:
                    policy[i][j] = pre_value
                    i += 1
                
                for i in range(v[k][0],v[k][1]):
                    policy[i][j]  = v[k][2]
                i += 1
            pre_value = v[-1][2]
            while i <= Tmax - 1:
                policy[i][j] = pre_value
                i += 1
        else:
            policy[:,j] = default_values[j]
                

        
    return policy"""
def get_policy(var):
    # var can be Str_l-- list of string used for make_policy
    # var can also be path used for read_policy
    if type(var) ==list:
        return make_policy(var)
    else:
        return read_policy(var)
    
def read_policy(path):
    df = pd.read_excel(path, header =0)
    l = []
    params = 3
    for i in range(params):
        l.append([])
    
        
    for j in range(params):
        for i in range(1,df.values.shape[0]):
        
        # policy 1
            if not isnan(df.values[i][3*j+1]):
                l[j].append((df.values[i][3*j+1]-1,df.values[i][3*j+2]))
            
            else:
                break
            

    T_max = l[0][-1][0] +1
    policy = np.zeros((T_max,params))
    for j in range(params):
        for k in range(len(l[j])-1):
            policy[l[j][k][0]:l[j][k+1][0],j] = l[j][k][1]
        
        policy[T_max-1,j] = l[j][k+1][1]
    return policy

def make_policy(Str_l):
    # used for input of mutiple of 2 return policy array 
    
    # Str_l is a list of decison string
    # eg. Str_l = ['1,0.2,10,0.4','20,0.6','17,0.8,30,0.9']
    # where Str_l[0] for a_sd means till day 1 use 0.2 ,till day 10 use 0.4
    # Str_l[1] for a_c means use 0.6 for all days (doesn't have to be '20', any value will have same effect)
    # Str_l[2] for a_u means till day 17 use 0.8 till day 30 use 0.9
    # in this case T_max(simulation days) will be 30 so for a_c from day 11 to day 30 will use 0.4
    # each string must be a mutiple of 2
    l = []
    params = len(Str_l)
    for i in range(params):
        l.append([])
    T_max = 0
    for j in range(len(Str_l)):
        Str = Str_l[j]
        L = Str.split(',')
        length = len(L) // 2
        
        if len(L) % 2!= 0:
            raise ValueError('input should be a mutiple of 2')
        
        for i in range(length):
            if int(L[2*i]) < 1:
                raise ValueError('day should be greater or equal to 1')
            l[j].append( (int(L[2*i])  , float(L[2*i + 1])) )
        if T_max < l[j][-1][0] :
            T_max = l[j][-1][0] 
        
    policy = np.empty((T_max,len(Str_l)))
    
    for j in range(params):
        if len(l[j]) == 1:
            policy[:,j] = l[j][0][1]
        
        else:
            policy[:l[j][0][0],j] = l[j][0][1]
            for k in range(1,len(l[j])):
                policy[l[j][k-1][0]:l[j][k][0],j] = l[j][k][1]
            policy[l[j][-1][0]:,j] = l[j][-1][1]
    
    return policy
            
"""if __name__ =='__main__':
    path = 'policy_example.xlsx'
    
    policy = read_policy(path)"""