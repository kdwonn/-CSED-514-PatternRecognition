import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

def getPrior(dataset):
    classes = dataset[:, [2]]
    prior_prob = np.zeros(2)
    num_dp = len(classes)

    for i in range(0, num_dp):
        dp_class = classes[i][0]
        prior_prob[int(dp_class)] += 1

    prior_prob /= num_dp
    return prior_prob

def getMeanVar(dataset):
    classes = dataset[:, [2]]
    points = dataset[:, [0, 1]]

    num_dp = len(classes)
    num_0 = np.count_nonzero(classes == 0)
    points_0 = np.zeros((num_0, 2))
    points_1 = np.zeros((num_dp - num_0, 2))

    temp0 = 0
    temp1 = 0
    for i in range(0, num_dp):
        if classes[i] == 0:
            points_0[temp0] = points[i]
            temp0 += 1
        else:
            points_1[temp1] = points[i]
            temp1 += 1
    
    means = [np.mean(points_0, axis=0), np.mean(points_1, axis=0)]
    var = np.cov(points_0.T)*(num_0/num_dp) + np.cov(points_1.T)*((num_dp - num_0)/num_dp)
    return means, var

def compLike(pi, m , v, test_data):
    prediction = np.zeros(test_data.shape[0])
    for i in range(0, test_data.shape[0]):
        like_0 = multivariate_normal(mean=m[0], cov=v).pdf(test_data[i]) * pi[0]
        like_1 = multivariate_normal(mean=m[1], cov=v).pdf(test_data[i]) * pi[1]
        probx = like_0 + like_1
        prob0 = like_0/probx
        prob1 = like_1/probx
        if prob0 > prob1 :
            prediction[i] = 0
        else:
            prediction[i] = 1
    return prediction

def predict(dataset, test):
    m, v = getMeanVar(dataset)
    pi = getPrior(dataset)

    test_label = test[:, [2]]
    test_data = test[:, [0, 1]]

    prediction = compLike(pi, m, v, test_data)

    accuracy = 0
    for i in range(0, prediction.shape[0]):
        if prediction[i] == test_label[i]:
            accuracy += 1

    return accuracy / prediction.shape[0] 

dataset = np.genfromtxt('two_moon.dat', skip_header=4)
print(dataset)
np.random.shuffle(dataset)
print(dataset[:, [2]])
print(dataset[:, [0, 1]].T)
trainset = dataset[0:1600]
testset = dataset[1600:2000]

print('acc : ')
print(predict(trainset, testset))