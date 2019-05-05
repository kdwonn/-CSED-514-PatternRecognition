from sklearn.neighbors import KernelDensity
import numpy as np
import math

def predict(trainset, testset):
    train_classes = trainset[:, [2]]
    train_points = trainset[:, [0, 1]]
    test_classes = testset[:, [2]]
    test_points = testset[:, [0, 1]]

    num_dp = len(train_classes)
    num_0 = np.count_nonzero(train_classes == 0)
    points_0 = np.zeros((num_0, 2))
    points_1 = np.zeros((num_dp - num_0, 2))
    
    prob0 = num_0/num_dp
    prob1 = (num_dp - num_0)/num_dp

    temp0 = 0
    temp1 = 0
    for i in range(0, num_dp):
        if train_classes[i] == 0:
            points_0[temp0] = train_points[i]
            temp0 += 1
        else:
            points_1[temp1] = train_points[i]
            temp1 += 1

    kde0 = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde0.fit(points_0)

    kde1 = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde1.fit(points_1)

    accuracy = 0
    for i in range(0, testset.shape[0]):
        prediction = 1
        probAnd0 = math.log(prob0) + kde0.score_samples(test_points[i].reshape(1,-1))
        probAnd1 = math.log(prob1) + kde1.score_samples(test_points[i].reshape(1,-1))
        if probAnd0 > probAnd1:
            prediction = 0
        if test_classes[i] == prediction:
            accuracy += 1
    
    return accuracy / testset.shape[0]

dataset = np.genfromtxt('two_moon.dat', skip_header=4)
np.random.shuffle(dataset)
trainset = dataset[0:1600]
testset = dataset[1600:2000]

print('acc : ')
print(predict(trainset, testset))
