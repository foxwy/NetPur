'''
@author: foxwy
@time: 2019-09-06
@description: node based extreme learning machine
'''

#/
#model
#/
import elm
from sklearn.model_selection import RepeatedKFold
import os
import numpy as np
from .source import *

#/
#极限学习机的训练
#function:sigmoid, sine, multiquadric, gaussian
#/
def ELM_node_train(X, Y, params):
    #params = [function, C, L, False]
    elmr = elm.ELMRandom(params)
    elmr._local_train(X, Y, params)

    # 返回计算结果
    return elmr

#/
#极限学习机的预测
#/
def ELM_node_predict(X, Y, elmr):
    return elmr._local_test(X, Y, predicting=False)

#--------------------------------------------------elm 参数选择--------------------------------------------------
#/
#the train processing of node based extreme learning machine using cross-validation method
#function:sigmoid, sine, multiquadric, gaussian
#/
def ELM_node_train_reKFold(indata, outdata, K, repeated_num, function):
    #----------ELM Init begin----------
    hidden_C = 2**-10
    test_error_aver_min = 1e10  #相对误差最小
    test_size = len(outdata) * len(outdata[0]) / K
    test_error_C_L = []  #保存每个C,L对应的误差

    #ELM参数
    C_best = 0
    L_best = 0

    #cross validation
    kf = RepeatedKFold(n_splits=K, n_repeats=repeated_num, random_state=12883823)

    #----------elm train begin----------
    print('\n-----node elm parameter selection-----')
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
                    elmr = ELM_node_train(Input_train, Output_train, [function, hidden_C, hidden_L, False])
                except:
                    exception_flag = 1
                    break
                else:
                    capacity_test = ELM_node_predict(Input_test, Output_test, elmr)  #elm测试
                    test_error_aver += np.linalg.norm(Output_test - capacity_test)  #矩阵F范数

            if exception_flag == 0:  #非异常
                test_error_C_L.append([hidden_C, hidden_L, (test_error_aver / (K * repeated_num))**2 / test_size])
                if test_error_aver_min > test_error_aver:  #更优参数
                    print('+++++++++++++++++++++++++')
                    test_error_aver_min = test_error_aver
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

    return C_best, L_best, test_error_aver_min, test_error_C_L

#--------------------------------------------------文件存储和提取--------------------------------------------------
#/
#存储node ELM参数
#/
def Store_node_elmdata(C_best, L_best, test_error_aver, test_error_C_L, save_num):
    #创建文件夹
    path = r'save\result'
    if not os.path.exists(path+'\\'+str(save_num)):
        os.mkdir(path+'\\'+str(save_num))
        os.mkdir(path+'\\'+str(save_num)+'\\'+'weights')
        os.mkdir(path+'\\'+str(save_num)+'\\'+'analysis')

    StoreWeights(C_best, path+'\\'+str(save_num)+r'\weights\C_best.txt')
    StoreWeights(L_best, path+'\\'+str(save_num)+r'\weights\L_best.txt')
    StoreWeights(test_error_aver, path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    StoreWeights(test_error_C_L, path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')

def Grab_node_elmdata(save_num):
    path = r'save\result'
    C_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\C_best.txt')
    L_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\L_best.txt')
    test_error_aver = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    test_error_C_L = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')

    return C_best, L_best, test_error_aver, test_error_C_L