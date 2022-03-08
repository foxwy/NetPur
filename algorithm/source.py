'''
@author: foxwy
@time: 2019-09-06
@description: machine learning basic function
'''
import numpy as np

#--------------------------------------------------文件存储和提取--------------------------------------------------
'''
@function: StoreWeights
@description: 将参数保存在电脑里
@input: data:待保存的数据,类型随意，FileName:保存成的文件名，诸如'XX.txt'
'''
def StoreWeights(data, FileName):
    import pickle
    fw = open(FileName, 'wb')
    pickle.dump(data, fw)
    fw.close()

'''
@function:GrabWeights
@description:将电脑保存的数据提取
@input:FileName:已保存的数据文件名，诸如'XX.txt'
@output:提取的保存文件
'''
def GrabWeights(FileName):
    import pickle
    fr = open(FileName, 'rb')
    return pickle.load(fr)

#--------------------------------------------------不同归一化方式--------------------------------------------------
'''
@function: normalization_global
@description: 全局归一化, 提取全局数据的最大最小值来归一化
@input: data:待归一化的数据，默认输入为二维矩阵或者列表, a, b:期望映射的范围[a, b], a, b大小未限定，一般a<b
@output: 归一化之后的数据，默认格式array
'''
def normalization_global(data, a, b):
    data = np.array(data)
    data_max = np.max(data)
    data_min = np.min(data)
    if data_max == data_min:
        data = (b - a) * data / data_max + a
    else:
        data = (b - a) * (data - data_min) / (data_max - data_min) + a
    return data

'''
@function: normalization_row
@description: 行归一化, 提取每行数据的最大最小值来归一化各自行
@input: data:待归一化的数据，默认输入为二维矩阵或者列表, a, b:期望映射的范围[a, b], a, b大小未限定，一般a<b
@output: 归一化之后的数据，默认格式array
'''
def normalization_row(data, a, b):
    data = np.array(data)
    data_norm = []
    for i in range(len(data)):
        row_max = np.max(data[i])
        row_min = np.min(data[i])
        if row_max == row_min:
            data_norm.append(((b - a) * data[i] / row_max + a).tolist())  #若直接赋值原矩阵的会强制转换导致错误
        else:
            data_norm.append(((b - a) * (data[i] - row_min) / (row_max - row_min) + a).tolist())  #若直接赋值原矩阵的会强制转换导致错误
    return np.array(data_norm)

'''
@function: normalization_column
@description: 列归一化, 提取每列数据的最大最小值来归一化各自列
@input: data:待归一化的数据，默认输入为二维矩阵或者列表, a, b:期望映射的范围[a, b], a, b大小未限定，一般a<b
@output: 归一化之后的数据，默认格式array
'''
def normalization_column(data, a, b):
    data = np.array(data)
    data = normalization_row(data.T, a, b).T
    return data

#/
#随机选择数据来作为训练集，测试集和预测集，移除部分数据
#/
def random_choose(train_num, test_num, predict_num, remove_nums):
    # 提取训练测试编号
    dataSet = list(range(train_num + test_num + predict_num + len(remove_nums)))
    for i in remove_nums:
        dataSet.remove(i)
    trainSet = []; testSet = []; predictSet = []  # 随机提取的数据

    for i in range(train_num):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        trainSet.append(dataSet[randIndex])
        del(dataSet[randIndex])

    for i in range(test_num):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        testSet.append(dataSet[randIndex])
        del(dataSet[randIndex])

    for i in range(predict_num):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        predictSet.append(dataSet[randIndex])
        del(dataSet[randIndex])
    return trainSet, testSet, predictSet

#--------------------------------------------------两向量相似度度量--------------------------------------------------
#https://www.jianshu.com/p/a7436ecd4240
#/
#曼哈顿距离，值越大，相关性越低
#/
def Manhattan_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    distance = np.sum(abs(x - y))
    return distance

#/
#欧式距离，值越大，相关性越低
#/
def Euclidean_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    distance = np.sqrt(np.sum((x - y)**2))
    return distance

#/
#皮尔逊相关系数，范围[-1, 1]，越接近0，相关性越低
#/
def PearsonCorrelation_coefficient(x, y):
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = n.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean)) #分子
    den = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))  #分母
    distance = num / den
    return distance

#/
#余弦距离
#/
def Cosin_similarity(x, y):
    x = no.array(x)
    y = np.array(y)

    num = np.sum(x * y)  #分子
    den = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))  #分母
    distance = num / den
    return distance

#/
#修正余弦距离
#/
def Cosin_adjust_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))  #分母
    distance = num / den
    return distance