import numpy as np
def getp_l_u(A_tp):
    N = A_tp.shape[0]
    P_t = np.zeros(N)
    L = np.zeros((N, N))
    P = np.zeros((N, N))
    for i in range(N):
        P_t[i] = i#index 设置初值
    for i in range(N - 1):

        index_max = np.argmax(abs(A_tp[i:N, i])) + i #找到当前主元index

        temp_P = P_t[index_max].copy()
        P_t[index_max] = P_t[i]
        P_t[i] = temp_P #index 交换

        temp_A = A_tp[index_max, :].copy()
        A_tp[index_max, :] = A_tp[i, :]
        A_tp[i, :] = temp_A #A矩阵两行交换

        temp_L = L[index_max, :].copy()
        L[index_max, :] = L[i, :]
        L[i, :] = temp_L #L矩阵两行交换

        for j in range(N - i - 1):
            L[j + 1 + i, i] = A_tp[j + 1 + i, i] / A_tp[i, i] #得到L参数
            A_tp[j + 1 + i, :] = A_tp[j + 1 + i, :] - A_tp[j + 1 + i, i] / A_tp[i, i] * A_tp[i, :] #高斯消元
        # print('L:', L)
        # print('A:', A)
        # print(index_max+i)
    for i in range(N):
        P[i, int(P_t[i])] = 1 #得到P矩阵
    L = L + np.eye(N) #得到L矩阵
    U = A_tp.copy() #U矩阵即消元后的A矩阵
    return (P, L, U)

def getinv_L(L):
    N = L.shape[0]
    L1 = np.eye(N)
    for i in range(N - 1):
        for j in range(i + 1):
            L1[i + 1, j] = -np.dot(L[i + 1, j:i + 1], L1[j:i + 1, j]) #根据矩阵乘法展开，迭代得到L逆矩阵的值
    return L1

def getinv_U(U):
    N = U.shape[0]
    dg = np.zeros((N, N))
    for i in range(N):
        dg[i, i] = 1.0 / U[i, i]  #U=P.dot(U_L) U1=U_L1.dot(dg) dg=P1,其对角元素为U对角元素的倒数
    U_L_T = U.copy()
    for i in range(N):
        U_L_T[i, :] = U_L_T[i, :] / U[i, i]  #这里将U_L转置，方便使用L求逆函数，
    return np.transpose(getinv_L(U_L_T.T)).dot(dg)
