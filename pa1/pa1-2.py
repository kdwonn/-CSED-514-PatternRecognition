import pandas as pd
import numpy as np
import math
from scipy.stats import multivariate_normal

def predict(trainset, testset, k):
    classes = trainset[:, [2]]
    points = trainset[:, [0, 1]]

    num_dp = len(classes)
    num_0 = np.count_nonzero(classes == 0)
    points_0 = np.zeros((num_0, 2))
    points_1 = np.zeros((num_dp - num_0, 2))

    prob0 = num_0/num_dp
    prob1 = (num_dp - num_0)/num_dp

    temp0 = 0
    temp1 = 0
    for i in range(0, num_dp):
        if classes[i] == 0:
            points_0[temp0] = points[i]
            temp0 += 1
        else:
            points_1[temp1] = points[i]
            temp1 += 1

    pi0, pi1, m0, m1, v0, v1 = initialize(points_0, points_1, k)
    pi0, m0, v0 = em(pi0, m0, v0, points_0)
    pi1, m1, v1 = em(pi1, m1, v1, points_1)

    test_classes = testset[:, [2]]
    test_points = testset[:, [0, 1]]

    accuracy = 0
    for i in range(0, testset.shape[0]):
        prediction = 0
        probAnd0 = prob0 * compGM(pi0, m0, v0, test_points[i])
        probAnd1 = prob1 * compGM(pi1, m1, v1, test_points[i])
        
        if probAnd0 > probAnd1:
            prediction = 0
        else:
            prediction = 1
        
        if prediction == test_classes[i]:
            accuracy += 1
    return accuracy / test_points.shape[0]

def initialize(points_0, points_1, k):
    pi0 = np.full(k, 1/k)
    pi1 = np.full(k, 1/k)
    m0 = np.random.randint(min(points_0[:,0]), max(points_0[:,0]), size=(k, 2))
    m1 = np.random.randint(min(points_1[:,0]), max(points_1[:,0]), size=(k, 2))
    # m1 = np.full((k, 2), np.mean(points_1, axis=0))
    v0 = np.zeros((k, 2, 2))
    v1 = np.zeros((k, 2, 2))
    for i in range(0, k):
        np.fill_diagonal(v0[i], 5)
        np.fill_diagonal(v1[i], 5)
    return pi0, pi1, m0, m1, v0, v1

def em(pi, m, v, points):
    k = len(pi)
    n = points.shape[0]

    resp = np.ones((n, k)) / k

    first_loop = True
    while True:
        n_in_k = np.ones(k)

        #calculate responsibility
        for i in range(0, n):
            sum = 0
            for j in range(0, k):
                sum += multivariate_normal(mean=m[j], cov=v[j]).pdf(points[i]) * pi[j]
            for j in range(0, k):
                resp[i][j] = multivariate_normal(mean=m[j], cov=v[j]).pdf(points[i]) * pi[j] / sum
        
        # calculate n_k
        n_in_k = np.sum(resp, axis=0)

        # cal m , v, pi
        pi = n_in_k / n
        m = np.matmul(resp.T, points) / n_in_k.T.reshape(k, 1)
        #print(m)
        v = np.zeros((k, 2, 2))
        for i in range(0, k):
            for j in range(0, n):
                tempm = points[j] - m[i]
                tempm.reshape(1, 2)
                # print(np.outer(tempm.T, tempm))
                v[i] += resp[j][i] * np.outer(tempm.T, tempm)
            v[i] /= n_in_k[i]
            # print(v[i])
            # print(resp)
            # print(m)

        if first_loop:
            log_like = compLogLike(pi, m, v, points)
            first_loop = False
        else:
            last_log_like = log_like
            log_like = compLogLike(pi, m, v, points)
            #print(log_like)
            if (log_like - last_log_like) * (log_like - last_log_like) < 1:
                break
    
    return pi, m, v

def compLogLike(pi, m, v, points):
    n = points.shape[0]
    k = len(pi)
    sum = 0
    for i in range(0, n):
        tempsum = 0
        for j in range(0, k):
            tempsum += pi[j] * multivariate_normal.pdf(points[i], mean=m[j], cov=v[j])
        sum += math.log(tempsum)
    return sum

def compGM(pi, m, v, point):
    k = len(pi)
    sum = 0
    for i in range(0, k):
        sum += pi[i] * multivariate_normal.logpdf(point, mean=m[i], cov=v[i])
    return sum


dataset = np.genfromtxt('two_moon.dat', skip_header=4)
#print(dataset)
np.random.shuffle(dataset)
#print(dataset[:, [2]])

trainset = dataset[0:1600]
testset = dataset[1600:2000]
print('acc (k = 3) : ')
score11 = predict(trainset, testset, 3)
print(score11)
print('acc (k = 5) : ')
score12 = predict(trainset, testset, 5)
print(score12)