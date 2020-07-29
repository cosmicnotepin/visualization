import math
from tabulate import tabulate

import globs

def printTLSRes(M):
    """
    TLS IV.1.3 (p. 106ff)
    """
    S = []
    for vc in list(globs.vClass)[0:9]:
        S.append(sum(M[vc.value]))

    A = []
    for vc in list(globs.vClass)[0:9]:
        A.append(S[vc.value]/sum(S))

    E1 = []
    for vc in list(globs.vClass)[0:9]:
        S_x = S[vc.value]
        if(S_x == 0):
            E1.append(0)
        else:
            E1.append(M[vc.value][vc.value]/S[vc.value])

    P_E1 = []
    for vc in list(globs.vClass)[0:9]:
        M_xx = M[vc.value][vc.value]
        Z = 1.96
        S_x = S[vc.value]
        if(S_x == 0):
            P_E1.append(0)
        else:
            P_E1.append((2*M_xx + Z**2 - Z*math.sqrt(Z**2 + 4*M_xx*(1 - M_xx/S_x)))/(2*(S_x + Z**2)))

    #normalize M for E2 calculations
    tlsProportions = [0, 0.007, 0.75, 0.06, 0.01, 0.05, 0.05, 0.07, 0.003]
    M_n = []
    M_n.append([0]*9)
    for vc in list(globs.vClass)[1:9]:
        c_norm = []
        for vci in list(globs.vClass)[0:9]:
            if(A[vc.value] ==0):
                c_norm.append(0)
            else:
                c_norm.append(M[vc.value][vci.value] * (tlsProportions[vc.value]/A[vc.value]))
        M_n.append(c_norm)
    
    S_n = []
    for vc in list(globs.vClass)[0:9]:
        S_n.append(sum(M_n[vc.value]))

    E2 = []
    for vc in list(globs.vClass)[0:9]:
        S_i = S_n[vc.value]
        if(S_i == 0):
            E2.append(0)
        else:
            SUM_M_xi = 0
            for x in range(0,9):
                SUM_M_xi += M_n[x][vc.value]

            M_ii = M_n[vc.value][vc.value]
            E2.append(1 - ((SUM_M_xi - M_ii)/S_i))

    P_E2 = []
    for vc in list(globs.vClass)[0:9]:
        SUM_M_xi = 0
        for x in range(0,9):
            SUM_M_xi += M_n[x][vc.value]
        M_ii = M_n[vc.value][vc.value]
        M_xx = SUM_M_xi - M_ii
        Z = 1.96
        S_x = S_n[vc.value]
        if(S_x == 0):
            P_E2.append(0)
        else:
            if(M_xx/S_x>1): # prints only to detect problems
                print(vc.name)
                print(x)
            P_E2.append(1 - (2*M_xx + Z**2 + Z*math.sqrt(Z**2 + 4*M_xx*(1 - M_xx/S_x)))/(2*(S_x + Z**2)))

    print("Abgesicherte Detektionsraten E1, E2")
    print(tabulate([P_E1, P_E2], headers = [vc.name for vc in globs.vClass][0:9]))
