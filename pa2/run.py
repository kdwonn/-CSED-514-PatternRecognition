import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def train_dict():
    temp = unpickle('data_batch_1')
    train_data = temp[b'data']
    train_labels = temp[b'labels']
    for i in range(2, 6):
        file_name = 'data_batch_'+ str(i)
        temp = unpickle(file_name)
        train_data = np.concatenate((train_data, temp[b'data']))
        train_labels = np.concatenate((train_labels, temp[b'labels']))
    return train_data, train_labels

def test_dict():
    temp = unpickle('test_batch')
    return temp[b'data'], temp[b'labels']

class KNN(object):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
    
    def eval(self, test_data, test_labels, k):
        n = test_data.shape[0]
        predictions = np.zeros(n)

        for i in range(0, n):
            nearest_k = np.zeros(k)
            dist = np.sum(np.abs(self.train_data - test_data[i]), axis = 1)
            sorted = np.argsort(dist)
            for j in range(0, k):
                nearest_k[j] = train_labels[sorted[j]]
            a = np.array(nearest_k)
            a = a.astype(int)
            most_voted = np.argmax(np.bincount(a))
            predictions[i] = most_voted
            print(most_voted == test_labels[i])

        return predictions

    def test(self, test_data, test_labels, k):
        prediction = self.eval(test_data, test_labels, k)
        accuracy = 0.0
        for i in range(0, prediction.shape[0]):
            if prediction[i] == test_labels[i]:
                 accuracy += 1
        accuracy = accuracy / prediction.shape[0]
        print('accuracy : '  + str(accuracy))

        return accuracy

class Linear(object):
    def __init__(self):
        pass
    def train(self, train_data, train_labels, epoch, lr, rr, test_data, test_labels):
        n = train_data.shape[0]
        img_dim = train_data.shape[1]
        self.weights = 0.001 * np.random.randn(10, img_dim)

        losses = np.zeros(epoch)
        val_accs = np.zeros(epoch)
        accs = np.zeros(epoch)
        
        for nth_epoch in range(0, epoch):
            # print(self.weights[0])
            val_train_data, val_test_data, val_train_labels, val_test_labels = train_test_split(train_data, train_labels, train_size = 0.005, test_size = 0.005, random_state = nth_epoch)
            
            # print(val_train_data.shape)
            # print(val_train_labels.shape)
            # print(val_test_data.shape)
            # print(val_test_labels.shape)

            n_train = val_train_data.shape[0]
            n_test = val_test_data.shape[0]

            # onehot_train_labels = np.zeros((n_train, 10))
            # onehor_test_labels = np.zeros((n_test, 10))
            # onehot_train_labels[np.arange(n_train), val_train_labels] = 1   
            # onehor_test_labels[np.arange(n_test), val_test_labels] = 1
            
            grad_w = np.zeros((10, img_dim))
            score = np.matmul(self.weights, val_train_data.T)
            # print(score.shape)
            print(score[val_train_labels, np.arange(n_train)].shape)
            loss = np.maximum(0, score - score[val_train_labels, np.arange(n_train)] + 1)
            loss[val_train_labels, np.arange(n_train)] = 0
            # print(loss[0])

            loss_with_regularizer = np.sum(loss) / n_train + rr * np.sum(self.weights * self.weights)
            losses[nth_epoch] = loss_with_regularizer

            ind = loss
            for i in range(0, loss.shape[0]):
                for j in range(0, loss.shape[1]):
                    if (ind[i][j] != 0):
                        ind[i][j] = 1

            ind[val_train_labels, np.arange(n_train)] = np.sum(ind, axis = 0) * (-1)

            grad_w = np.dot(ind, val_train_data) / n_train + rr * 2 * self.weights
            self.weights -= grad_w * lr
            # print(self.weights[0])

            val_prediction = self.eval(val_test_data)
            val_accuracy = 0.0
            for i in range(0, val_prediction.shape[0]):
                if val_prediction[i] == val_test_labels[i]:
                     val_accuracy += 1
            val_accuracy = val_accuracy / val_prediction.shape[0]
            val_accs[nth_epoch] = val_accuracy
            print('validation acc for' + str(nth_epoch) + 'th epoch :' + str(val_accuracy))

            real_prediction = self.eval(test_data)
            real_accuracy = 0.0
            for i in range(0, real_prediction.shape[0]):
                if real_prediction[i] == test_labels[i]:
                     real_accuracy += 1
            real_accuracy = real_accuracy / real_prediction.shape[0]
            accs[nth_epoch] = real_accuracy

        return losses, val_accs, accs
        
    def eval(self, test_data):
        prediction = np.zeros(test_data.shape[0])
        score = np.matmul(self.weights, test_data.T)
        prediction = np.argmax(score, axis = 0).reshape(test_data.shape[0])

        return prediction

    def test(self, test_data, test_labels):
        prediction = self.eval(test_data)
        accuracy = 0.0
        for i in range(0, prediction.shape[0]):
            if prediction[i] == test_labels[i]:
                 accuracy += 1
        accuracy = accuracy / prediction.shape[0]
        print('accuracy : '  + str(accuracy))

        return accuracy






train_data, train_labels = train_dict()
test_data, test_labels = test_dict()

lc = Linear()
lc_losses, lc_val_accs, lc_accs = lc.train(train_data, train_labels, 400, 1e-7, 1e4, test_data, test_labels)
lc_accuracy = lc.test(test_data, test_labels)

x = np.arange(400)
plt.plot(x, lc_val_accs, label = 'accuracy : validation')
plt.plot(x, lc_accs, label = 'accuracy : test')
plt.xlabel("epoch")
plt.legend()
plt.show()

for i in range(0, 10):
    plt.subplot(10, 1, i+1)
    weight3d = lc.weights[i].reshape((32, 32, 3))
    weight2d = np.mean(weight3d, axis = 2)
    plt.matshow(weight2d)
    # plt.imshow(lc.weights[i].reshape((32, 32, 3)).astype('uint8'), vmin = 0, vmax = 255)
plt.show()

knn = KNN(train_data, train_labels)
knn5_acc = knn.test(test_data, test_labels, 5) 