# -*- coding: utf-8 -*-

#from math import log, exp
import numpy as np
from numpy import imag, power, log, exp, e
import sys

import math

#from read_data import read_HPfit, read_conf
#from funcs_derivs import trans, S, K_h, dfS_x_thSR, dKh_x , dKT_x , THETA , K_t, dfS_x_triad, dfS_x_delta  ### importa funcoes e derivadas com "step" imaginario
#from funcs_derivs import prepare_data_ths21 , prepare_data_thsSR, prepare_data_dual, prepare_data_triad


import re



#### Path and File name  ##########

path = r'.' + '\\'


file_data = 'data.hpf'
conf_data = 'configuration.cfg'



def read_HPfit(path , file, logK , log_base ):
    '''
    Read function of HP_fit.
    Necessita término, apenas leitura dos dados para testes.
    
    retorna duas listas
    -lista com dados de potencial e de teor de água (ordem: h e theta)
    -lista com pesos, na ordem dos dados inseridos
    
    '''
    
    location = path + file
    
    datahT = []
    datahK = []
    dataTK = []
    
    weighthT = []
    weighthK = []
    weightTK = []
    
    with open(location, 'r') as f:
        
        fff = [i.split() for i in f.readlines()]
        fff = [i for i in fff if i != [] ]   # remove listas vazias
        
    hT, hK, TK = False, False, False
    for ii in fff:
        
        if ii == ['*end', 'of', 'file']:
            break
        
        # hT
        if ii == ['*', 'h', 'theta', 'weight']:
            hT = True
            hK, TK = False, False
            continue
        if hT and (ii == ['*', 'h', 'K', 'weight'] or ii == ['*', 'theta', 'K', 'weight']):
            hT = False
        # hK
        if ii == ['*', 'h', 'K', 'weight']:
            hK = True
            hT, TK = False, False
            continue
        if hK and (ii == ['*', 'h', 'theta', 'weight'] or ii == ['*', 'theta', 'K' , 'weight']):
            hK = False
        # TK
        if ii == ['*', 'theta', 'K', 'weight' ]:
            TK = True
            hT, hK = False, False
            continue
        if TK and (ii == ['*', 'h', 'K', 'weight'] or ii == ['*', 'h', 'theta', 'weight']):
            TK = False
        
        if hT:
            datahT.append([ float(ii[0]) , float(ii[1]) ])
            if len(ii) <= int(2):
                weighthT.append( int(1) )
            else:
                weighthT.append( float(ii[2]) )
        if hK:
            if logK:
                datahK.append([ float(ii[0]) , log(float(ii[1]))/log(log_base)   ])
            else:
                datahK.append([ float(ii[0]) , float(ii[1]) ])
            if len(ii) <= int(2):
                weighthK.append( int(1))
            else:
                weighthK.append( float(ii[2]) )
        if TK:
            if logK:
                dataTK.append([ float(ii[0]) , log( float(ii[1])) /log(log_base) ])
            else:
                dataTK.append([ float(ii[0]) , float(ii[1]) ])
            if len(ii) <= int(2):
                weightTK.append( int(1) )
            else:
                weightTK.append( float(ii[2]) )
        
    return datahT, weighthT, datahK, weighthK, dataTK, weightTK


#w = read_HPfit(path , file)

def read_conf (path, file ):
    with open(path + file) as f:
        
        ff = [re.split(r' |:|;|,|=|\n|\t|_|\?|\/', i) for i in f.readlines()]
        ff = [list(filter(None ,i)) for i in ff]
        ff = [i for i in ff if i != [] ]

    cfg = {}
    for i, var in enumerate(ff):
        
        if var[0].lower() == 'version':
            cfg['version'] = ff[i + 1][0].lower()
        if var[0].lower() == 'model':
            cfg['model'] = ff[i + 1][0].lower()
        if var[0].lower() == 'data' and var[1].lower() == 'type':
            cfg['select'] = tuple([ii.lower() for ii in  ff[i + 1]])
        
        if var[0].lower() == 'fit' and var[1].lower() == 'initial' and var[2].lower() == 'lower':
            cfg['fit']     = []
            cfg['par_ini'] = []
            cfg['low_lim'] = []
            cfg['up_lim']  = []
            
            for ii in range(7):
                if ff[ii + i + 1][0].lower() == 'y' or ff[ii + i + 1][0].lower() == '1':
                    cfg['fit'].append(True)
                    
                if ff[ii + i + 1][0].lower() != 'y' and ff[ii + i + 1][0].lower() != '1':
                    cfg['fit'].append(False)
                
                cfg['par_ini'].append(float( ff[ii + i + 1][1]  ))
                cfg['low_lim'].append(float( ff[ii + i + 1][2]  ))
                cfg['up_lim'].append( float( ff[ii + i + 1][3]  ))
            cfg['par_ini'].append(float( ff[7 + i + 1][1]  ))
            cfg['par_ini'].append(float( ff[8 + i + 1][1]  ))
            
        if var[0].lower() == 'maximum' and var[1].lower() == 'number' and var[3].lower() == 'iterations':
            cfg['max_iter'] = int(round(  float(ff[i + 1][0])  ))
        if var[0].lower() == 'print' and var[1].lower() == 'each' and var[2].lower() == 'step':
            cfg['p_step'] = float( ff[i + 1][0])
        if var[0].lower() == 'transformation' and var[2].lower() == 'parameters':
            if ff[i + 1][0].lower() == 'y':
                cfg['tran'] = 1
            else:
                cfg['tran'] = 0
        if var[0].lower() == 'log' and var[1].lower() == 'k' and var[2].lower() == 'instead':
            if ff[i + 1][0].lower() == 'y':
                cfg['logK'] = True
            else:
                cfg['logK'] = False
        if var[0].lower() == 'minimum' and var[1].lower() == 'data' and var[2].lower() == 'mix' and var[3].lower() == 'deviation':
            cfg['mix_d'] = float( ff[i + 1][0])
        if var[0].lower() == 'delta' and var[1].lower() == 'rmsd':
            if ff[i + 1][0].lower() == 'y':
                cfg['delta_chi'] = True
                cfg['delta_chi_val'] = float(ff[i + 1][1])
            else:
                cfg['delta_chi'] = False
                cfg['delta_chi_val'] = 0
        if var[0].lower() == 'parameter' and var[1].lower() == 'change':
            if ff[i + 1][0].lower() == 'y':
                cfg['par_change_c'] = True
                cfg['par_change'] = float(ff[i + 1][1])
            else:
                cfg['par_change_c'] = False
                cfg['par_change'] = 0        
        if var[0].lower() == 'initial' and var[1].lower() == 'ml' and var[2].lower() == 'damp':
            cfg['ml_ini'] = float( ff[i + 1][0])
        if var[0].lower() == 'ml' and var[1].lower() == 'multiplier':
            cfg['ml_mul'] = float( ff[i + 1][0])
        if var[0].lower() == 'log' and var[1].lower() == 'k' and var[2].lower() == 'base':
            if ff[i + 1][0].lower() == 'e':
                cfg['base_K'] = e
            else:
                cfg['base_K'] = float( ff[i + 1][0])
        
        if var[0].lower() == 'weight' and var[1].lower() == 'k' and var[2].lower() == 'data':
            if ff[i + 1][0].lower() == 'y':
                cfg['weight_data'] = True
            else:
                cfg['weight_data'] = False
        
    return cfg


def trans(var, x , method = 'f' ):
    '''methods: forward, f ; or backwards, b'''
    
    if method == 'f':
        if var == 'a':
            A = log(x)
            return A
        if var == 'n':
            N  = log(x - 1)
            return N
        if var == 'Ks':
            Kss = log(x)
            return Kss
        if var == 'l':
            l = x
            return l
        if var == 'delta_SR':
            delta = x
            return delta
        if var == 'thS' or var == 'thR' or var == 'th2' or var == 'th1':
            return x
    if method == 'b':
        if var == 'a':
            a  = exp(x)
            return a
        if var == 'n':
            n  = 1 + exp(x) 
            return n
        if var == 'Ks':
            Ks = exp(x)
            return Ks
        if var == 'l':
            l = x
            return l
        if var == 'delta_SR':
            delta = x
            return delta
        if var == 'thS' or var == 'thR' or var == 'th2' or var == 'th1':
            return x

# Relative water content function
def S(h, a, n, tran ):
    ''' If tran = 1, transformation will be applied
    '''
    # VGM
    if abs(h) < 1e-6:
        return 1
    
    if tran == 1:
        a = trans('a', x = a, method = 'b' )
        n = trans('n', x = n, method = 'b')

    ff2 = power( (1 + power((a * h), n) ) , ( - 1 + 1/n)  )
        
    return ff2

def THETA(th, thS, thR):
    
    TH = (th - thR)/(thS - thR)
    if isinstance(TH, complex):
        print("Problem with convergence of a parameter")
        sys.exit()

    if TH < 0:
        TH = 0
    return TH

# Conductivity in function of h
def K_h(f, h, a, n, Ks , l, tran ):
    # VGM
    if tran == 1:
        a  = trans('a',  x =  a, method = 'b')
        n  = trans('n',  x =  n, method = 'b')
        Ks = trans('Ks', x = Ks, method = 'b')
        l  = trans('l',  x =  l, method = 'b')
        
    ff3 = Ks * power(f(h, a, n, tran=0), l) * power( 1- power((  1 - power(f(h, a, n, tran=0) , (n / (n-1)))) , (1 -1/n))   , 2)
        
    #else:
    #    ff = Ks * power(f(h, a, n, tran), l) * power( 1- power((  1 - power(f(h, a, n, tran) , (n / (n-1)))) , (1 -1/n))   , 2)
    
    return ff3

# Conductivity in function of Theta
def K_t(f, th, n, Ks, l, thS, thR, tran):
    # VGM
    
    if tran == 1:
        n  = trans('n',  x =  n, method = 'b')
        Ks = trans('Ks', x = Ks, method = 'b')
        l  = trans('l',  x =  l, method = 'b')
    
    ff4 = Ks * power(f(th, thS, thR), l) * power( 1- power((  1 - power(f(th, thS, thR) , (n / (n-1)))) , (1 -1/n))   , 2)
    
    #print(th, thS, thR)
    #ff = Ks * power(f(th, thS, thR) , l) * power((1- power((1-    power(f(th, thS, thR) ,(n/(n-1)) )    )  , (1-(1/n)))  ) ,2)
    
    #ff4 = Ks * f(th, thS, thR) ** l *(1- (1- f(th, thS, thR) **(n/(n-1)) )**(1-(1/n))  )**2
    
    return ff4

### -------------------------------------------------------------------------------------------------------
# Derivatives of Th

def dfS_x_th21(f, h, a, n, var, tran, t2, t1, h2,h1,  hh = 1e-8):
    
    if var == 'a':
        a = a + 1j * hh
    
    if var == 'n':
        n = n + 1j * hh
    
    if var == 'th2':
        t2 = t2 + 1j * hh
    
    if var == 'th1':
        t1 = t1 + 1j * hh
    
    f = t2 - (t2 - t1) *  ((f(h2,a,n,tran) - f(h,a,n,tran)) / (f(h2,a,n,tran) - f(h1,a,n,tran)))
    
    ret = imag( f )/hh
    return ret

def dfS_x_thSR(f, h, a, n, var, tran, thS, thR,  hh = 1e-8):
    
    if var == 'a':
        a = a + 1j * hh
    
    if var == 'n':
        n = n + 1j * hh

    if var == 'thS':
        thS = thS + 1j * hh
        
    if var == 'thR':
        thR = thR + 1j * hh
    
    f = thR + (thS - thR) *  f(h,a,n,tran)
    
    ret = imag( f )/hh
    return ret

def dfS_x_delta(f, h, a, n, h1, t1, var, tran, delta_th = 0 , hh = 1e-8):
    
    if var == 'a':
        a = a + 1j * hh
    
    if var == 'n':
        n = n + 1j * hh
        
    if var == 'h':
        h  = h  + 1j * hh
        h1 = h1 + 1j * hh
    
    if var == 'delta_SR':
        delta_th = delta_th + 1j * hh
    
    f = t1 - delta_th *  (f(h1,a,n,tran) - f(h,a,n,tran))
    ret = imag( f )/hh
    return ret

def dfS_x_triad(f, h, a, n, h1, h2 , t1, t2 , var, tran , hh = 1e-8):
    
    if var == 'a':
        a = a + 1j * hh
    
    if var == 'n':
        n = n + 1j * hh
        
    if var == 'h':
        h  = h  + 1j * hh
        h1 = h1 + 1j * hh
        h2 = h2 + 1j * hh
    
    if var == 'th2':
        t2 = t2 + 1j * hh
    
    if var == 'th1':
        t1 = t1 + 1j * hh
        
    f = t2 - (t2 - t1) *  ((f(h2,a,n,tran) - f(h,a,n,tran)) / (f(h2,a,n,tran) - f(h1,a,n,tran)))
    ret = imag( f )/hh
    return ret


### -------------------------------------------------------------------------------------------------------
# Derivatives of Kh


def dKh_x(f, h, a, n, Ks , l , f2, var , tran , hh = 1e-8 ):
    
    if var == 'a':
        a = a + 1j * hh
    
    if var == 'n':
        n = n + 1j * hh
        
    if var == 'Ks':
        Ks = Ks + 1j * hh
        
    if var == 'l':
        l = l + 1j * hh
        
    if var == 'h':
        h = h + 1j * hh
    
    k = f(f2, h, a, n, Ks , l, tran ) 
    ret = imag( k )/hh
       
    return ret


### -------------------------------------------------------------------------------------------------------
# Derivatives of KT

def dKT_x(f, th, n, Ks , l, thS, thR, f2, var , tran , hh = 1e-8):

    if var == 'th':
        th = th + 1j * hh
    
    if var == 'thS':
        thS = thS + 1j * hh
    
    if var == 'thR':
        thR = thR + 1j * hh

    if var == 'n':
        n = n + 1j * hh
    
    if var == 'Ks':
        Ks = Ks + 1j * hh
    
    if var == 'l':
        l = l + 1j * hh
    

    k = f(f2, th, n, Ks, l, thS, thR, tran)

    ret = imag( k )/hh
    
    return ret








### -------------------------------------------------------------------------------------------------------

def prepare_data_ths21(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d):
    med_hT, med_hK, med_TK = 0, 0, 0
    if 'hT'.lower() in select:
        for i in range(len(hT)):
            datahT.append([hT[i], whT[i]])
            weighthT.append(whT[i])
            # WEIGHT
            med_hT += hT[i][1]

        N += len(datahT)
        med_hT = abs(med_hT/len(hT))
    
    datahK = []
    weighthK = []
    med_hK = 0
    if 'hK'.lower() in select:
        for i in range(len(hK)):
            datahK.append([hK[i], whK[i]])
            weighthK.append(whK[i])
            # WEIGHT
            med_hK += hK[i][1]

        N += len(datahK)
        med_hK = abs(med_hK/len(hK))
    
    dataTK = []
    weightTK = []
    med_TK = 0
    if 'TK'.lower() in select:
        for i in range(len(TK)):
            dataTK.append([TK[i], wTK[i]])
            weightTK.append(wTK[i])
            # WEIGHT
            med_TK += TK[i][1]

        N += len(dataTK)
        med_TK = abs(med_TK/len(TK))
    return datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N



def prepare_data_thsSR(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d):    
    med_hT, med_hK, med_TK = 0, 0, 0
    if 'hT'.lower() in select:
        for i in range(len(hT)):
            datahT.append([hT[i], whT[i]])
            weighthT.append(whT[i])
            # WEIGHT
            med_hT += hT[i][1]

        N += len(datahT)
        med_hT = abs(med_hT/len(hT))
    
    datahK = []
    weighthK = []
    if 'hK'.lower() in select:
        for i in range(len(hK)):
            datahK.append([hK[i], whK[i]])
            weighthK.append(whK[i])
            # WEIGHT
            med_hK += hK[i][1]

        N += len(datahK)
        med_hK = abs(med_hK/len(hK))
    
    dataTK = []
    weightTK = []
    if 'TK'.lower() in select:
        for i in range(len(TK)):
            dataTK.append([TK[i], wTK[i]])
            weightTK.append(wTK[i])
            # WEIGHT
            med_TK += TK[i][1]

        N += len(dataTK)
        med_TK = abs(med_TK/len(TK))
        
    return datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N


def prepare_data_dual(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d):
    pool_l = 2
    med_hT, med_hK, med_TK = 0, 0, 0
    if 'hT'.lower() in select:
        for i in range(len(hT)):
            i_glob = i
            # WEIGHT
            med_hT += hT[i][1]
            for ii in range(len(hT)):
                if i == ii :
                    continue  # th 0   th 1
                if (abs(hT[i][1] - hT[ii][1]) < mix_d):   # Removing uncompatible data
                    continue
                else:
                    # weight adjustment
                    if (i_glob == i and i!= 0):
                        for iv in range(len(pool)):
                            pool[iv][pool_l] = pool[iv][pool_l]/len(pool)
                            weighthT.append(pool[iv][pool_l])
                        datahT = datahT + pool
                        pool = []
                    pool.append([hT[i], hT[ii], whT[i]])
                    i_glob += 1
        
        for iv in range(len(pool)):
            pool[iv][pool_l] = pool[iv][pool_l]/len(pool)
            weighthT.append(pool[iv][pool_l])
        datahT = datahT + pool

        N += len(datahT)
        med_hT = abs(med_hT/len(hT))
    
    datahK = []
    weighthK = []
    if 'hK'.lower() in select:
        for i in range(len(hK)):
            datahK.append([hK[i], whK[i]])
            weighthK.append(whK[i])
            # WEIGHT
            med_hK += hK[i][1]

        N += len(datahK)
        med_hK = abs(med_hK/len(hK))
    
    dataTK = [] # removed TK data
    weightTK = []
    N += len(dataTK)
    med_TK = 1
    
    
    return datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N

def prepare_data_triad(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d, step = 1):
    
    if step >= 2:
        3
        ## UTILIZAR VALORES nas bordas como fixos e então realizar a regressão, como no Excel.
    
    
    
    
    
    if step == 0:
        pool_l = 3
        med_hT, med_hK, med_TK = 0, 0, 0
        if 'hT'.lower() in select:
            for i in range(len(hT)):
                # WEIGHT
                med_hT += hT[i][1]
                i_glob = i
                for ii in range(len(hT)):
                    for iii in range(len(hT)):
                        if i == ii or i == iii or ii >= iii:   # ii >= iii para não repetir theta1 e theta2
                            continue  # th 0   th 1    # th 2
                        if (abs(hT[i][1] - hT[ii][1]) < mix_d)  or  (abs(hT[i][1] - hT[iii][1]) < mix_d):   # Removing uncompatible data
                            continue
                        else:
                            if (i_glob == i and i!= 0):  
                                # weight adjustment
                                for iv in range(len(pool)):
                                    pool[iv][pool_l] = pool[iv][pool_l]/len(pool)
                                    weighthT.append(pool[iv][pool_l])
                                datahT = datahT + pool
                                pool = []
                            pool.append([hT[i], hT[ii], hT[iii], whT[i]])
                            i_glob += 1
            
            for iv in range(len(pool)):
                pool[iv][pool_l] = pool[iv][pool_l]/len(pool)
                weighthT.append(pool[iv][3])
            datahT = datahT + pool

            N += len(datahT)
            med_hT = abs(med_hT/len(hT))
        
        datahK = []
        weighthK = []
        if 'hK'.lower() in select:
            for i in range(len(hK)):
                # WEIGHT
                med_hK += hK[i][1]
                datahK.append([hK[i], whK[i]])
                weighthK.append(whK[i])
            N += len(datahK)
            med_hK = abs(med_hK/len(hK))
        
        dataTK = [] # removed TK data
        weightTK = []
        N += len(dataTK)
        med_TK = 1
    
    
    ####  NEXT VALUE APPROACH #### BAD!
    
    if step == 1:
        med_hT, med_hK, med_TK = 0, 0, 0
        
        if 'hT'.lower() in select:
            # Sort maintaining weight order
            for i, val in enumerate(hT):
                val.append(whT[i])
            hT.sort()
            whT2 = []
            for i, val in enumerate(hT):
                whT2.append(hT[i][2])
                hT[i].pop(2)
            
            # STACK START -----------
            stack_hT = []
            ind = 0
            ver = False
            skip = False
            for i in range(len(hT)):
                if (i + 1) == len(hT):
                    if not ver:
                        stack_hT.append( [ hT[i] ] )
                    continue
                
                
                if abs(hT[i][0] - hT[i + 1][0]) < mix_d or abs(hT[i][1] - hT[i + 1][1]) < 0.001 :
                    if ver:
                        stack_hT[ind].append( hT[i + 1] )
                    else:
                        ver = True
                        stack_hT.append( [] )
                        stack_hT[ind] =  [hT[i] , hT[i + 1]]
                        skip = True
                else:
                    ind += 1
                    ver = False
                    if skip:
                        skip = False
                        continue
                    else:
                        stack_hT.append( [ hT[i] ] )
            # STACK END -----------
            #for i in stack_hT:
            #    print(i)
            #sys.exit()
            # Combining data START --------------------------------------
            ind = 0
            for i in range(len(stack_hT)):
                if i == 0:
                    for ii in range(len(stack_hT[i])):
                        for iii in stack_hT[i + 1]:
                            for iv in stack_hT[i + 2]:
                                datahT.append([stack_hT[i][ii], iii, iv, whT2[ind]])
                                weighthT.append(whT2[ind])
                        ind += 1
                
                elif (i + 1) == len(stack_hT):
                    for ii in range(len(stack_hT[i])):
                        for iii in stack_hT[i - 1]:
                            for iv in stack_hT[i - 2]:
                                datahT.append([stack_hT[i][ii], iii, iv, whT2[ind]])
                                weighthT.append(whT2[ind])
                        ind += 1
                
                else:
                    for ii in range(len(stack_hT[i])):
                        for iii in stack_hT[i - 1]:
                            for iv in stack_hT[i + 1]:
                                datahT.append([stack_hT[i][ii], iii, iv, whT2[i]])
                                weighthT.append(whT2[ind])
                        ind += 1
            
            # Combining data END --------------------------------------
            
            # Weight for K calculation
            for i in range(len(hT)):
                med_hT += hT[i][1]
            med_hT = abs(med_hT/len(hT))
                        
            N += len(datahT)
            

        datahK = []
        weighthK = []
        if 'hK'.lower() in select:
            for i in range(len(hK)):
                # WEIGHT
                med_hK += hK[i][1]
                datahK.append([hK[i], whK[i]])
                weighthK.append(whK[i])
            N += len(datahK)
            med_hK = abs(med_hK/len(hK))
        
        dataTK = [] # removed TK data
        weightTK = []
        N += len(dataTK)
        med_TK = 1
        
    return datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N


##### Read configuration file ######
cfg = read_conf (path, conf_data)

version, model , select , mix_d = cfg['version'], cfg['model'], cfg['select'], cfg['mix_d']
max_runs , transf , ini_ML_damp = cfg['max_iter'], cfg['tran'], cfg['ml_ini']
delta_sum, delta_sum_c = cfg['delta_chi_val'], cfg['delta_chi']
par_change, par_change_c = cfg['par_change'], cfg['par_change_c']
logK, p_step, ML_mult,  base_K = cfg['logK'], cfg['p_step'], cfg['ml_mul'], cfg['base_K']
fit, par_ini, low_lim, up_lim = cfg['fit'], cfg['par_ini'], cfg['low_lim'], cfg['up_lim']

weight_data = cfg['weight_data']

hh2 = par_ini[7]
hh1 = par_ini[8]

theta22 = par_ini[4]
theta11 = par_ini[5]
nn = par_ini[1]
aa = par_ini[0]


##### Read data from data file #######
hT, whT, hK, whK, TK, wTK = read_HPfit(path, file_data, logK = logK, log_base = base_K)


if version == 'triad':
    par_total = 4
    par_str = ['a', 'n', 'Ks', 'l']
    par_ini = par_ini[:4]
    fit     = fit[:4]
    low_lim = low_lim [:4]
    up_lim  = up_lim[:4]
    
if version == 'dual':
    par_total = 5
    par_str = ['a', 'n', 'Ks', 'l', 'delta_SR']
    par_ini = par_ini[:4]  + [par_ini[6]]
    fit     = fit[:4]      + [fit[6]]
    low_lim = low_lim[:4]  + [low_lim[6]]
    up_lim  = up_lim[:4]   + [up_lim[6]]
    
if version == 'ths-thr':
    par_total = 6
    par_str = ['a', 'n', 'Ks', 'l', 'thS', 'thR']
    par_ini = par_ini[:6]
    fit     = fit[:6]
    low_lim = low_lim[:6]
    up_lim  = up_lim[:6]
    
if version == 'th2-th1':
    par_total = 6
    par_str = ['a', 'n', 'Ks', 'l', 'th2', 'th1']
    par_ini = par_ini[:6]
    fit     = fit[:6]
    low_lim = low_lim[:6]
    up_lim  = up_lim[:6]
    h2 = cfg['par_ini'][7]
    h1 = cfg['par_ini'][8]
    
    if 'tk' in select:
        print('Do not support "th2-th1" version with "Theta-K" data')
        sys.exit()

len_fit = 5    ##### DADOS DE CADA PARAMETRO.

############## ARRUMARR !!!!!!!!! #########################################################
#fit_ini_par = [ [0.1, True, 'a', 0.0001, 10] , [4, True, 'n', 1.01, 5] , [3.0, True, 'Ks', 0.001, 60] , [0.5, True, 'l', -5, 6] ]

fit_ini_par = [[] for i in range(par_total)]  ############################# ARRUMAR!!!! ##################################


for i in range(par_total):    
    for ii in range(len_fit):
        fit_ini_par[i].append(0)

for i in range(par_total):
    
    fit_ini_par[i][0] = par_ini[i]
    fit_ini_par[i][1] = fit[i]
    fit_ini_par[i][3] = low_lim[i]
    fit_ini_par[i][4] = up_lim[i]
    
    fit_ini_par[i][2] = par_str[i]

if transf == 1 :
    print('Transformations applied')

#print(fit_ini_par)

N = int(0)        # Number of "observed points" (considerin the mixing of data and transformations)
M = int(0)        # Number of parameters to be fitted

####################################################################################################
# Functions of derivatives of h and K. Maintain the in the same order as fit_ini_par.
#             ['a',         'n',        'Ks',   'l',    'thS',        'thR',         delta]
if version == 'th2-th1':
    deri_h  = [dfS_x_triad, dfS_x_triad, 0,     0,     dfS_x_triad, dfS_x_triad]
    deri_K  = [dKh_x,      dKh_x,      dKh_x, dKh_x, 0,          0]
    deri_KT = [0,          dKT_x,      dKT_x, dKT_x, dKT_x,      dKT_x]
if version == 'ths-thr':
    deri_h  = [dfS_x_thSR, dfS_x_thSR, 0,     0,     dfS_x_thSR, dfS_x_thSR]
    deri_K  = [dKh_x,      dKh_x,      dKh_x, dKh_x, 0,          0]
    deri_KT = [0,          dKT_x,      dKT_x, dKT_x, dKT_x,      dKT_x]
if version == 'dual':
    deri_h  = [dfS_x_delta, dfS_x_delta, 0,     0,     dfS_x_delta]
    deri_K  = [dKh_x,       dKh_x,       dKh_x, dKh_x,           0]
    deri_KT = []
if version == 'triad':
    deri_h  = [dfS_x_triad, dfS_x_triad,  0,      0]
    deri_K  = [dKh_x,       dKh_x,      dKh_x,   dKh_x]
    deri_KT = []
####################################################################################################


# Transforms
if transf == 1:
    for i in range(len(fit_ini_par)):
        var = fit_ini_par[i][2]
        fit_ini_par[i][0] = trans(var, x = fit_ini_par[i][0])
        fit_ini_par[i][3] = trans(var, x = fit_ini_par[i][3])
        fit_ini_par[i][4] = trans(var, x = fit_ini_par[i][4])

# Capturando os chutes iniciais
ini = []       # valores dos parâmetros iniciais e string com os nomes
par_fit = []
for i in fit_ini_par:
    if i[1]:
        ini.append((i[0],i[2]))
        par_fit.append(i[0])
M = len(ini)

datahT = []
weighthT = []
pool = []

### PREPARING DATA - Mixing data and calculating weights for K  ###### 
if version == 'th2-th1':
    datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N  = prepare_data_ths21(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d)
if version == 'ths-thr':
    datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N  = prepare_data_thsSR(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d)
if version == 'dual':
    datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N  = prepare_data_dual(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d)
if version == 'triad':
    datahT, weighthT, datahK, weighthK, dataTK, weightTK, med_hT, med_hK, med_TK, N  = prepare_data_triad(select , datahT, weighthT, hT, whT, hK, whK, TK, wTK ,N , pool, mix_d)

# Applying Weight to K
if weight_data:
    if 'hT'.lower() in select:
        if 'hK'.lower() in select:
            for i in range(len(datahK)):
                datahK[i][-1] = datahK[i][-1] * med_hT/med_hK
                weighthK[i] = weighthK[i] * med_hT/med_hK
        if 'TK'.lower() in select:
            for i in range(len(dataTK)):
                dataTK[i][-1] = dataTK[i][-1] * med_hT/med_TK
                weightTK[i] = weightTK[i] * med_hT/med_TK


N_weight = len(hT) + len(hK) + len(TK)  # N observed points (not mixed)
weight = weighthT + weighthK + weightTK   # appending weight lists
jac = np.zeros((N, M))      # Jacobian Matrix
W = np.diag(weight)         # Weight matrix
res = np.zeros((N))
I = np.identity((M)) 

#par = [fit_ini_par[0][0], fit_ini_par[1][0], fit_ini_par[2][0], fit_ini_par[3][0] ] # Necessita os 6 na ordem correta!!!
par = []
for i in range(len(fit_ini_par)):
    par.append(fit_ini_par[i][0])

print('Selected version:', version)
print('Selected model:', model )




for iii in range(max_runs):
    
    # Defining: Jacobiano e Residuo
    if version == 'th2-th1':
        a   = par[0]
        n   = par[1]
        Ks  = par[2]
        l   = par[3]
        t2  = par[4]
        t1  = par[5]

    if version == 'ths-thr':
        a   = par[0]
        n   = par[1]
        Ks  = par[2]
        l   = par[3]
        thS = par[4]
        thR = par[5]
        
    if version == 'dual':
        a  = par[0]
        n  = par[1]
        Ks = par[2]
        l  = par[3]
        delta = par[4]
    if version == 'triad':
        a  = par[0]
        n  = par[1]
        Ks = par[2]
        l  = par[3]
    
    index_N = 0
    if 'hT'.lower() in select:
        for ii in range(len(datahT)):
            if version == 'th2-th1':
                #res
                h  =     datahT[ii][0][0]
                th_obs = datahT[ii][0][1]
                
                th_calc = t2 - (t2 - t1) * (S(h2,a,n,tran = transf) - S(h, a, n,tran = transf)) / (S(h2,a,n,tran = transf) - S(h1,a,n,tran = transf))
                
                res[index_N] = th_obs - th_calc 
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_h[i] != 0:
                            var = fit_ini_par[i][2]
                            jac[index_N][index] = deri_h[i](S, h, a, n, h1, h2 , t1, t2 , var, tran = transf)
                        else:
                            jac[index_N][index] = 0
                        index += 1                    
                index_N += 1
            if version == 'ths-thr':
                #res
                h  =     datahT[ii][0][0]
                th_obs = datahT[ii][0][1]
                
                th_calc = thR + (thS - thR)  * S(h,a,n,tran = transf)
                
                res[index_N] = th_obs - th_calc 
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_h[i] != 0:
                            var = fit_ini_par[i][2]
                            jac[index_N][index] = deri_h[i](S, h, a, n, var, tran = transf, thS = thS, thR = thR)
                        else:
                            jac[index_N][index] = 0
                        index += 1                    
                index_N += 1
            
            if version == 'dual':
                #res
                h  =     datahT[ii][0][0]
                th_obs = datahT[ii][0][1]
                h1 =     datahT[ii][1][0]
                t1 =     datahT[ii][1][1]
                
                th_calc = t1 - delta * ( S(h1,a,n,tran = transf) - S(h,a,n,tran = transf) )
                
                res[index_N] = th_obs - th_calc 
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_h[i] != 0:
                            var = fit_ini_par[i][2]
                            jac[index_N][index] = deri_h[i](S, h, a, n, h1,  t1, var, tran = transf, delta_th = delta)
                        else:
                            jac[index_N][index] = 0  
                        index += 1                    
                index_N += 1
            if version == 'triad':
                #res
                th_obs = datahT[ii][0][1]
                h  =     datahT[ii][0][0]
                h1 =     datahT[ii][1][0]
                h2 =     datahT[ii][2][0]
                t1 =     datahT[ii][1][1]
                t2 =     datahT[ii][2][1]
                
                th_calc = t2 - (t2 - t1) * (S(h2,a,n,tran = transf) - S(h, a, n,tran = transf)) / (S(h2,a,n,tran = transf) - S(h1,a,n,tran = transf))
                
                res[index_N] = th_obs - th_calc 
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_h[i] != 0:
                            var = fit_ini_par[i][2]
                            jac[index_N][index] = deri_h[i](S, h, a, n, h1, h2 , t1, t2, var, tran = transf)
                        else:
                            jac[index_N][index] = 0
                        index += 1                    
                index_N += 1
                

    if 'hK'.lower() in select:
        for ii in range(len(datahK)):
            if version == 'th2-th1':
                # res
                h     = datahK[ii][0][0]
                K_obs = datahK[ii][0][1]
                
                K_calc = K_h(S, h, a, n, Ks , l, tran = transf)
                K_calc2 = K_calc
                
                if logK:
                    K_calc = np.log(K_calc) / np.log(base_K)
                
                res[index_N] = (K_obs - K_calc)
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_K[i] != 0:
                            var = fit_ini_par[i][2]
                            if logK:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf) / (np.log(base_K) * K_calc2)
                            else:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf)
                                
                        if deri_K[i] == 0:
                            jac[index_N][index] = 0
                        index += 1
                index_N += 1
            if version == 'ths-thr':
                # res
                h     = datahK[ii][0][0]
                K_obs = datahK[ii][0][1]
                
                K_calc = K_h(S, h, a, n, Ks , l, tran = transf)
                K_calc2 = K_calc
                
                if logK:
                    K_calc = np.log(K_calc) / np.log(base_K)
                
                res[index_N] = (K_obs - K_calc)
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_K[i] != 0:
                            var = fit_ini_par[i][2]
                            if logK:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf) / (np.log(base_K) * K_calc2)
                            else:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf)
    
                        if deri_K[i] == 0:
                            jac[index_N][index] = 0
                        index += 1
                index_N += 1
            if version == 'dual':

                # res
                h  = datahK[ii][0][0]
                K_obs = datahK[ii][0][1]
                
                K_calc = K_h(S, h, a, n, Ks , l, tran = transf)
                K_calc2 = K_calc
                
                if logK:
                    K_obs = K_obs / np.log(base_K)
                    K_calc = np.log(K_calc) / np.log(base_K)
                
                res[index_N] = (K_obs - K_calc)
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_K[i] != 0:
                            var = fit_ini_par[i][2]
                            if logK:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf) / (np.log(base_K) * K_calc2) 
                            else:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf)
                        if deri_K[i] == 0:
                            jac[index_N][index] = 0
                        index += 1
                index_N += 1
                
                
            if version == 'triad':            
                # res
                h  = datahK[ii][0][0]
                K_obs = datahK[ii][0][1]
                
                K_calc = K_h(S, h, a, n, Ks , l, tran = transf)
                K_calc2 = float(K_calc)
                
                if logK:
                    K_obs = K_obs / np.log(base_K)
                    K_calc = np.log(K_calc) / np.log(base_K)
                
                res[index_N] = (K_obs - K_calc)
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_K[i] != 0:
                            var = fit_ini_par[i][2]
                            if logK:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf) / (np.log(base_K) * K_calc2)  
                            else:
                                jac[index_N][index] =  deri_K[i](K_h, h, a, n, Ks , l  , S, var, tran = transf)
                        if deri_K[i] == 0:
                            jac[index_N][index] = 0
                        index += 1
                index_N += 1
        
    if 'TK'.lower() in select:
        for ii in range(len(dataTK)):
            if version == 'th2-th1':
                print('Do not support version "th2-th1" with "Theta - K" data')
                sys.exit()
                # res
                th  = dataTK[ii][0][0]
                K_obs = dataTK[ii][0][1]
    
                K_calc = K_t(THETA, th, n, Ks, l, thS, thR, tran = transf)
                K_calc2 = float(K_calc)
                
                if logK:
                    K_calc = np.log(K_calc) / np.log(base_K)
                
                res[index_N] = (K_obs - K_calc)
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        if deri_KT[i] != 0:
                            var = fit_ini_par[i][2]
                            if logK:
                                jac[index_N][index] =  deri_KT[i](K_t,  th, n, Ks , l, thS, thR, THETA, var, tran = transf) / (np.log(base_K) * K_calc2)
                            else:
                                jac[index_N][index] =  deri_KT[i](K_t,  th, n, Ks , l, thS, thR, THETA, var, tran = transf)
                        if deri_KT[i] == 0:
                            jac[index_N][index] = 0
                        index += 1
                index_N += 1
            
            
            if version == 'ths-thr':
                # res
                th  = dataTK[ii][0][0]
                K_obs = dataTK[ii][0][1]
                
                K_calc = K_t(THETA, th, n, Ks, l, thS, thR, tran = transf)
                K_calc2 = float(K_calc)
                                
                if logK:
                    K_calc = np.log(K_calc) / np.log(base_K)
                res[index_N] = (K_obs - K_calc)
                
                #jac
                index = 0
                for i in range(len(par)):
                    if fit_ini_par[i][1]:
                        
                        if deri_KT[i] != 0:
                            var = fit_ini_par[i][2]
                            
                            if logK:
                                jac[index_N][index] =  deri_KT[i](K_t,  th, n, Ks , l, thS, thR, THETA, var, tran = transf) / (np.log(base_K) * K_calc2)
                            else:
                                jac[index_N][index] =  deri_KT[i](K_t,  th, n, Ks , l, thS, thR, THETA, var, tran = transf)
                            
                        if deri_KT[i] == 0:
                            jac[index_N][index] = 0
                        
                        index += 1
                index_N += 1
   
            if version == 'dual':
                print('Theta-K data not supported in this version')
                sys.exit()
            if version == 'triad':
                print('Theta-K data not supported in this version')
                sys.exit()
    
    hess =  np.linalg.multi_dot( [jac.T, W ,jac] )
    if iii == 0:
        ML_damp = ini_ML_damp * max(np.diagonal(hess))
    hess2 = hess + ML_damp * I

    inver = np.linalg.inv(hess2)
    delta_a = np.linalg.multi_dot( [inver , jac.T , W , res ] )
    
    ###  Max step   -------------------------------------------------------- change for better solution
    if transf == 1 :
        for cou in range(len(delta_a)):
            if delta_a[cou] > 0.9:
                delta_a[cou] = 0.9
                
            if delta_a[cou] < -0.9:
                delta_a[cou] = -0.9
            
            if math.isnan(delta_a[cou]):
                delta_a[cou] = 0.0
                print('Problem with convergence of a parameter, (parameter change = NAN)')
                sys.exit()
        
    if transf == 0 :
        for cou in range(len(delta_a)):
            if delta_a[cou] > 5:
                delta_a[cou] = 5
                
            if delta_a[cou] < -5:
                delta_a[cou] = -5
            
            if math.isnan(delta_a[cou]):
                delta_a[cou] = 0.0
                print('Problem')
                sys.exit()

    
    # Substituting parameters. Limiting maximum step.
    index = 0
    for i in range(len(par)):
        if fit_ini_par[i][1]:
            if (par[i] + delta_a[index]) <= fit_ini_par[i][3]:
                par[i] = fit_ini_par[i][3]
            elif (par[i] + delta_a[index]) >= fit_ini_par[i][4]:
                par[i] = fit_ini_par[i][4]
            else:
                par[i] = par[i] + delta_a[index]
            index += 1
    
    res_sq = 0
    resX2_2 = 0
    resX2_hT = 0
    resX2_hK = 0
    resX2_TK = 0
    residual = np.zeros((N))
    for i in range(N):
        res_sq += np.power( res[i]  , 2) * W[i][i]    ####### VERIFIACR QUAL É O CORRETO!!!
        
        # Generating X2_2
        if version == 'th2-th1':
            a   = par[0]
            n   = par[1]
            Ks  = par[2]
            l   = par[3]
            t2  = par[4]
            t1  = par[5]
            thS = 0
            thR = 0
        if version == 'ths-thr':
            a   = par[0]
            n   = par[1]
            Ks  = par[2]
            l   = par[3]
            thS = par[4]
            thR = par[5]
        if version == 'dual':
            a  = par[0]
            n  = par[1]
            Ks = par[2]
            l  = par[3]
            delta = par[4]
        if version == 'triad':
            a  = par[0]
            n  = par[1]
            Ks = par[2]
            l  = par[3]
        
                
        if version == 'th2-th1':
            if (i + 1) <= len(datahT):   #    hT DATA
                ind = i
                
                th_obs = datahT[ind][0][1]
                h      = datahT[ind][0][0]
                
                th_calc2 = t2 - (t2 - t1) * (S(h2,a,n,tran = transf) - S(h, a, n,tran = transf)) / (S(h2,a,n,tran = transf) - S(h1,a,n,tran = transf))
                
                dum = th_obs - th_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hT += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            

            
            elif (i + 1) <= (len(datahT) + len(datahK)):  #  hK DATA
                ind = i - len(datahT)
                
                h     = datahK[ind][0][0]
                K_obs = datahK[ind][0][1]
                
                K_calc2 = K_h(S, h, a, n, Ks , l, tran = transf)
                
                if logK:
                    K_calc2 = np.log(K_calc2) / np.log(base_K)
                 
                dum = K_obs - K_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hK += np.power( dum  , 2) * W[i][i]
                
                residual[i] = dum * W[i][i]
            
            elif(i + 1) <= (len(datahT) + len(datahK) + len(dataTK)):   # TK DATA
                ind = i - len(datahT) - len(datahK)
                
                th  = dataTK[ii][0][0]
                K_obs = dataTK[ii][0][1]
                
                K_calc2 = K_t(THETA, th, n, Ks, l, thS, thR, tran = transf)
                
                if logK:
                    K_calc2 = np.log(K_calc2) / np.log(base_K)
                
                dum = K_obs - K_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_TK += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
    
        if version == 'ths-thr':
            
            if (i + 1) <= len(datahT):   #    hT DATA
                ind = i
                
                th_obs = datahT[ind][0][1]
                h      = datahT[ind][0][0]
                
                th_calc2 = thR + (thS - thR)  * S(h,a,n,tran = transf)
                
                dum = th_obs - th_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hT += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
            elif (i + 1) <= (len(datahT) + len(datahK)):  #  hK DATA
                ind = i - len(datahT)
                
                h     = datahK[ind][0][0]
                K_obs = datahK[ind][0][1]
                
                K_calc2 = K_h(S, h, a, n, Ks , l, tran = transf)
                
                if logK:
                    K_calc2 = np.log(K_calc2) / np.log(base_K)
                 
                dum = K_obs - K_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hK += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
            elif(i + 1) <= (len(datahT) + len(datahK) + len(dataTK)):   # TK DATA
                ind = i - len(datahT) - len(datahK)
                
                th  = dataTK[ii][0][0]
                K_obs = dataTK[ii][0][1]
                
                K_calc2 = K_t(THETA, th, n, Ks, l, thS, thR, tran = transf)
                
                if logK:
                    K_calc2 = np.log(K_calc2) / np.log(base_K)
                
                dum = K_obs - K_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_TK += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
                
        if version == 'dual':
            
            if (i + 1) <= len(datahT):     #    hT DATA
                ind = i
                
                th_obs = datahT[ind][0][1]
                h  =     datahT[ind][0][0]
                h1 =     datahT[ind][1][0]
                t1 =     datahT[ind][1][1]
                
                th_calc2 = t1 - delta * (S(h1,a,n, tran = transf) - S(h, a, n, tran = transf)) 
                
                dum = th_obs - th_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hT += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
            elif (i + 1) <= (len(datahT) + len(datahK)):  #  hK DATA
                ind = i - len(datahT)
                
                h  = datahK[ind][0][0]
                K_obs = datahK[ind][0][1]
                
                K_calc2 = K_h(S, h, a, n, Ks , l, tran = transf)
                
                if logK:
                    K_obs = K_obs / np.log(base_K)
                    K_calc2 = np.log(K_calc2) / np.log(base_K)
                
                dum = K_obs - K_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hK += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
            elif(i + 1) <= (len(datahT) + len(datahK) + len(dataTK)):     # TK DATA
                ind = i - len(datahT) - len(datahK)
                print('Problem (TK data and model Dual)')
                sys.exit()
            
        if version == 'triad':    
    
            if (i + 1) <= len(datahT):       #    hT DATA
                ind = i
                
                th_obs = datahT[ind][0][1]
                h  =     datahT[ind][0][0]
                h1 =     datahT[ind][1][0]
                h2 =     datahT[ind][2][0]
                t1 =     datahT[ind][1][1]
                t2 =     datahT[ind][2][1]
                
                th_calc2 = t2 - (t2 - t1) * (S(h2,a,n, tran = transf) - S(h, a, n, tran = transf)) / (S(h2,a,n, tran = transf) - S(h1,a,n, tran = transf))
                dum = th_obs - th_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hT += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
            elif (i + 1) <= (len(datahT) + len(datahK)):    #  hK DATA
                ind = i - len(datahT)
                
                h  = datahK[ind][0][0]
                K_obs = datahK[ind][0][1]
                
                K_calc2 = K_h(S, h, a, n, Ks , l, tran = transf)
                
                if logK:
                    K_obs = K_obs / np.log(base_K)
                    K_calc2 = np.log(K_calc2) / np.log(base_K)
                
                dum = K_obs - K_calc2
                resX2_2 += np.power( dum  , 2) * W[i][i]
                resX2_hK += np.power( dum  , 2) * W[i][i]
                residual[i] = dum * W[i][i]
            
            elif(i + 1) <= (len(datahT) + len(datahK) + len(dataTK)):    # TK DATA
                ind = i - len(datahT) - len(datahK)
                print('Problem (TK data and model Triad)')
                sys.exit()
    
    
    ### Removido -M de N_weight -----------------
    varian  = res_sq /(N_weight)
    varian2 = resX2_2/(N_weight)
    
    resX2_hT2 = resX2_hT/(N_weight)
    resX2_hK2 = resX2_hK/(N_weight)
    resX2_TK2 = resX2_TK/(N_weight)
    
    X2 = res_sq
    X2_2 = resX2_2
    
    if ML_damp != 0:
        if X2_2 >= X2:
            ML_damp = ML_damp * ML_mult
        if X2_2 < X2:
            ML_damp = ML_damp / ML_mult
            # Stop criteria
            if (abs(varian - varian2) < delta_sum) and (delta_sum_c) and (iii > 5):
                print('Stop criteria reached "delta RMSD" reached')
                break

    if (iii + 1) % p_step == 0:
        print('run:', iii + 1)
        print('parameters:', par)
        print('last_step:', delta_a)
        print('ML_damper:', ML_damp)
        print('X2 - X2_2 :', (X2 - X2_2))
        print()

    # Stop criteria
    val = 0
    if par_change_c:
        for i in delta_a:
            if abs(i) < abs(par_change) and (iii > 5):
                val += 1
        if val >= len(delta_a):
            print('Stop criteria "minimum parameter change" reached')
            break


# DELTA COEFICIENT ###########################################################################################
desv = np.linalg.inv(hess)  *  (np.linalg.multi_dot([residual.T , residual]) / (N_weight-M))
cor = np.zeros(shape=(desv.shape[0], desv.shape[1]))
for i in range(desv.shape[0]):
    for ii in range(desv.shape[1]):
        cor[i][ii] = desv[i][ii] / (abs(desv[i][i] * desv[ii][ii] ) ** 0.5  )
index = 0
step_lis = []
for i in range(len(fit_ini_par)):
    if fit_ini_par[i][1]:
        step = desv[index][index] ** 0.5
        step_lis.append(step)
        index += 1

# De-transformation  #########################################################################################
upper = []
lower = []
CoV   = []
dev = []
if transf == 1:
    index = 0
    for i in range(len(fit_ini_par)):
        if fit_ini_par[i][1]:
            up  = par[i] + step_lis[index]
            dow = par[i] - step_lis[index]
            var = fit_ini_par[i][2]
            upper.append( trans(var, x = up, method = 'b') )
            lower.append( trans(var, x = dow, method = 'b') )
            CoV.append(abs(par[i] / step_lis[index]))
            temp = par[i]
            par[i] = trans(var, x = par[i], method = 'b')
            dev.append([ var , par[i], temp, step_lis[index] ])
            index += 1
        
else:
    index = 0
    for i in range(len(fit_ini_par)):
        if fit_ini_par[i][1]:
            up  = par[i] + step_lis[index]
            dow = par[i] - step_lis[index]
            var = fit_ini_par[i][2]
            upper.append( up  )
            lower.append( dow )
            CoV.append(abs(par[i] / step_lis[index]))
            dev.append([ var , par[i], -9999, step_lis[index] ])
            index += 1

if version == 'dual':

    thS = 0
    thR = 0
    sum_w = 0
    a  = par[0]
    n  = par[1]
    delta = par[4]
    for i in range(len(datahT)):
        th1 = datahT[i][1][1]
        h1 =  datahT[i][1][0]
        S1 =  S(h1, a, n, tran = 0 )
        
        th0 = datahT[i][0][1]
        h0 =  datahT[i][0][0]
        S0 =  S(h0, a, n, tran = 0 )
        
        wgh = datahT[i][2]  #########   ARRUMAR, remover necessidade de utiliozar datahT no loop
        sum_w += wgh     
        
        thS += wgh * (th0*(1 - S1) - th0*(1 - S0)) / (S0-S1)
        thR += wgh * (th0*S1 - th1*S0) / (S1-S0)
    
    thS = thS / sum_w
    thR = thR / sum_w
    thS2 = thR + delta
    
    
if version == 'triad':
    # Calculus of ThetaS and ThetaR
    thS = 0
    thR = 0
    sum_w = 0
    a  = par[0]
    n  = par[1]
    for i in range(len(datahT)):
        th1 = datahT[i][1][1]
        th2 = datahT[i][2][1]
        h1 =  datahT[i][1][0]
        h2 =  datahT[i][2][0]
        S1 =  S(h1, a, n, tran = 0 )
        S2 =  S(h2, a, n, tran = 0 )
        wgh = datahT[i][3]
        sum_w += wgh     
        
        thS += wgh * (th2*(1 - S1) - th1*(1 - S2)) / (S2-S1)
        thR += wgh * (th1*S2 - th2*S1) / (S2-S1)
    thS = thS / sum_w
    thR = thR / sum_w


if transf == 1 :
    transf2 = 'Transformations APPLIED'
    transf3 = '\nStatistics made before the "de-transformations"'
else:
    transf2 = 'Transformations NOT applied'
    transf3 = ''

print('RESULTS  -- ', transf2, transf3)
print('No. iterations:', iii + 1)

print('Fitted parameters:')
print(r"      Fitted value   Upper limit (95%)    Lower limit (95%)    **Deviation +/-")
index = 0
for i in range(len(fit_ini_par)):
    if fit_ini_par[i][1]:
        step = desv[index][index] ** 0.5
        print(f'{fit_ini_par[i][2]:4}' , ':', f'{par[i]:9.5e}', '|', f'{upper[index]:11.7e}',
              '    | ', f'{lower[index]:11.7e}', "    |" , f"{dev[index][3]:14.7E}" )
        
        if fit_ini_par[i][2] == 'th2':
            theta22 = par[i]
        if fit_ini_par[i][2] == 'th1':
            theta11 = par[i]
        if fit_ini_par[i][2] == 'n':
            nn = par[i]
        if fit_ini_par[i][2] == 'a':
            aa = par[i]

        index += 1


if transf == 1:
    print("** deviation calculated from transformed parameters alpha and n")

else:
    print("**unstransformed deviation")

if version == 'th2-th1':
    print()
    #print(hh2, hh1)
    #print(aa, nn, theta22, theta11)
    
    thetaS = theta22*(1 - S(hh1, aa, nn, 0 )) - theta11*(1 - S(hh2, aa, nn, 0 ))
    thetaS = thetaS / (S(hh2, aa, nn, 0 ) - S(hh1, aa, nn, 0 ))
    
    thetaR = theta11 * S(hh2, aa, nn, 0 ) - theta22 * S(hh1, aa, nn, 0 )
    thetaR = thetaR /  (S(hh2, aa, nn, 0 ) - S(hh1, aa, nn, 0 ))
    print('thS: ', thetaS)
    print('thR: ', thetaR)
    

if version == 'dual':

    #print('ThS:', round(thS, 5)  , ' Consider using thS delta')
    print('ThS delta:', round(thS2, 5))
    print('ThR:', round(thR, 5))
    print('ThS delta:', round(thS2, 5))
if version == 'triad':
    print('ThS:', round(thS, 5))
    print('ThR:', round(thR, 5))

print()
print('Correlation Matrix:')
print('      ', end='')
param= []
for i in fit_ini_par:
    if i[1]:
        print( f'{i[2]:10s}' , end = '' )
        param.append(i[2])

print()
for i in range(len(cor)):
    print(f'{param[i]:3s}', end = '')
    for ii in range(len(cor[i])):
        if ii > i :
            continue
        print( f'{cor[i][ii]:9.5f}'  , end=' ')
    print()

print()

print(f'SSD_hT: {resX2_hT2:.7E} ; RMSD_hT: {resX2_hT2**0.5:.7E}')
print(f'SSD_hK: {resX2_hK2:.7E} ; RMSD_hK: {resX2_hK2**0.5:.7E}')
print(f'SSD_TK: {resX2_TK2:.7E} ; RMSD_TK: {resX2_TK2**0.5:.7E}')
print(f'SSD: {varian2:.7f} ; RMSD: {varian2**0.5:.7f}')

if weight_data:
    if 'hK'.lower() in select:
        print('Weight K [hK]:' , f'{med_hT/med_hK:.5f}')
    if 'TK'.lower() in select:
        print('Weight K [TK]:' , f'{med_hT/med_TK:.5f}')

'''
print()
if transf == 1:
    print("TRANSFORMED VALUES")

    print(r"Parameter = untransformed parameter values  ")
    for i in dev:
        print(f'{i[0]:3}', "=",  i[2], )
'''

sys.exit()















