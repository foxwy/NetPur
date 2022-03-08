'''
@author: foxwy
@time: 2019-09-06
@description: extreme learning machine
'''

#/
#model
#/
import numpy as np
from sklearn.model_selection import RepeatedKFold
import os

from .source import *

#/
#sigmoid函数
#/
def sigmoid(x, w, b):
    '''
    定义Sigmoid函数: g(z) = 1 / (1 + e^-(xw + b))
    '''
    return 1.0 / (1 + np.exp(-1.0 * (x.dot(w) + b)))

#/
#极限学习机的训练
#/
def ELM_train(X, Y, C, L):
    # 随机初始化
    w = np.random.normal(size=(len(X[0]), L))
    b = np.random.normal(size=L)

    # 使用特征映射求解输出矩阵
    H = sigmoid(X, w, b)

    # 计算输出权重和输出函数
    if L <= len(X):
        beta = np.linalg.pinv(H.T.dot(H)+np.identity(L)/C).dot(H.T.dot(Y))
    else:
        beta = H.T.dot(np.linalg.pinv(H.dot(H.T)+np.identity(len(X))/C).dot(Y))

    # 返回计算结果
    return w, b, beta

#/
#极限学习机的预测
#/
def ELM_predict(X, w, b, beta):
    H = sigmoid(X, w, b)
    return H.dot(beta)

#--------------------------------------------------elm 参数选择--------------------------------------------------
#/
#the train processing of extreme learning machine
#/
def ELM_train_reKFold(indata, outdata, K, repeated_num):
    #----------ELM Init begin----------
    hidden_C = 2**-10
    test_error_aver_min = 1e10  #相对误差最小
    test_size = len(outdata) * len(outdata[0]) / K
    test_error_C_L = []  #保存每个C,L对应的误差

    #ELM参数
    w = 0
    b = 0
    beta = 0
    w_best = 0
    b_best = 0
    beta_best = 0
    C_best = 0
    L_best = 0

    #cross validation
    kf = RepeatedKFold(n_splits=K, n_repeats=repeated_num, random_state=12883823)

    #----------elm train begin----------
    print('\n-----sig_elm parameter selection-----')
    while hidden_C < 2**10:
        hidden_L = 100

        #-----inside train begin-----
        while hidden_L <= 4000:
            test_error_aver = 0
            exception_flag = 0

            for train_index, test_index in kf.split(indata):
                Input_train = indata[train_index]
                Output_train = outdata[train_index]
                Input_test = indata[test_index]
                Output_test = outdata[test_index]

                try:  #求逆异常
                    w, b, beta = ELM_train(Input_train, Output_train, C=hidden_C, L=hidden_L)  #elm训练
                except:
                    exception_flag = 1
                    break
                else:
                    capacity_test = ELM_predict(Input_test, w, b, beta)  #elm测试
                    test_error_aver += np.linalg.norm(Output_test - capacity_test)  #矩阵F范数

            if exception_flag == 0:  #非异常
                test_error_C_L.append([hidden_C, hidden_L, (test_error_aver / (K * repeated_num))**2 / test_size])
                if test_error_aver_min > test_error_aver:  #更优参数
                    print('+++++++++++++++++++++++++')
                    test_error_aver_min = test_error_aver
                    w_best = w
                    b_best = b
                    beta_best = beta
                    C_best = hidden_C
                    L_best = hidden_L

                print(hidden_L)
            hidden_L += 100
        #-----inside train end-----

        hidden_C *= 2
        print('error', (test_error_aver_min / (K * repeated_num))**2 / test_size)
        print('\n-----------------hidden_C-----------------')
        print(hidden_C)
    #----------elm train end----------
    test_error_aver_min = (test_error_aver_min / (K * repeated_num))**2 / test_size  #预测均方误差

    return w_best, b_best, beta_best, C_best, L_best, test_error_aver_min, test_error_C_L

#--------------------------------------------------文件存储和提取--------------------------------------------------
#/
#存储ELM参数
#/
def Store_sig_elmdata(w_best, b_best, beta_best, C_best, L_best, test_error_aver, test_error_C_L, save_num):
    #创建文件夹
    path = r'save\result'
    if not os.path.exists(path+'\\'+str(save_num)):
        os.mkdir(path+'\\'+str(save_num))
        os.mkdir(path+'\\'+str(save_num)+'\\'+'weights')
        os.mkdir(path+'\\'+str(save_num)+'\\'+'analysis')

    StoreWeights(w_best, path+'\\'+str(save_num)+r'\weights\w_best.txt')
    StoreWeights(b_best, path+'\\'+str(save_num)+r'\weights\b_best.txt')
    StoreWeights(beta_best, path+'\\'+str(save_num)+r'\weights\beta_best.txt')
    StoreWeights(C_best, path+'\\'+str(save_num)+r'\weights\C_best.txt')
    StoreWeights(L_best, path+'\\'+str(save_num)+r'\weights\L_best.txt')
    StoreWeights(test_error_aver, path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    StoreWeights(test_error_C_L, path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')

def Grab_sig_elmdata(save_num):
    path = r'save\result'
    w_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\w_best.txt')
    b_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\b_best.txt')
    beta_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\beta_best.txt')
    C_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\C_best.txt')
    L_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\L_best.txt')
    test_error_aver = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    test_error_C_L = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')

    return w_best, b_best, beta_best, C_best, L_best, test_error_aver, test_error_C_L