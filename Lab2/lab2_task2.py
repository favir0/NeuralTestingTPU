from keras.datasets import mnist
import numpy as np


# Считывание данных
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = {}
    data['images'] = np.array(X_train, dtype='float32').tolist()
    data['labels'] = np.array(y_train, dtype='float32').tolist()

    testData = {}
    testData['images'] = np.array(X_test, dtype='float32').tolist()
    testData['labels'] = np.array(y_test, dtype='float32').tolist()

    print("Количество изображений в обучающей выборке: ", len(X_train))
    return data, testData

class Neuron():
    def __init__(self, weights, value_to_find, learning_rate, inp):
        self.weights = weights.copy()
        self.value_to_find = value_to_find.copy()
        self.learning_rate = learning_rate
        self.inp = inp.copy()
    
    def solve_task(self, inpIndex):
        # Текущее значение функции
        curAnswer = 0
        for i in range(len(self.weights)):
            curAnswer += self.weights[i] * self.inp[inpIndex][i]
    
        # --------------- Пороговая функция активации ---------------
        if curAnswer >= 1: 
            curAnswer = 1
        else:
            curAnswer = 0
        # -----------------------------------------------------------
        return curAnswer 

    def recalc_weights(self, valueIndex, curAnswer):
        for i in range(len(self.weights)):
            self.weights[i] -= self.calc_grad_with_mse(self.inp[valueIndex][i], curAnswer, self.value_to_find[valueIndex])

    def calc_grad_with_mse(self, currentInp, curRes, promisedRes):
        d = self.calc_derivative()
        grad_down = -2 * (promisedRes-curRes) * d * currentInp * self.learning_rate
        return grad_down
 
    def calc_derivative(self):
        return 1

    def train_neuron(self):
        complete_train = False
        max_point = 1000
        while not complete_train:
            allAnswersRight = True
            for i, result in enumerate(self.value_to_find):
                # Текущее значение функции
                curAnswer = self.solve_task(i)
                #print("curAnswer = ", curAnswer)
                if result != curAnswer:
                    allAnswersRight = False
                    self.recalc_weights(i, curAnswer)
            max_point -=1
            if (max_point == 0):
                print("Inf train!")
                break
            complete_train = allAnswersRight

def specify_data(specific_label):
    train_data, test_data = load_data()
    print(len(train_data['labels']))
    print(len(train_data['images']))
    weights = np.zeros(len(train_data['labels']))
    specified_data = np.zeros(len(train_data['labels']))
    for i, data_label in enumerate(train_data['labels']):
        weights[i] = 0.1
        if data_label == specific_label:
            specified_data[i] = 1
        else:
            specified_data[i] = 0
    return specified_data, weights
        

def main():
    train_data, test_data = load_data()
    spec_data, w = specify_data(0)
    someData = spec_data[0:300]
    weights = w[0:300]
    new_train_data = []

    for image in train_data['images']:
        res = []
        for x in image:
            res.extend(x if isinstance(x, list) else [x])
        new_train_data.append(res)
    print(new_train_data[0])
    print(len(new_train_data[0]))
    print(len(new_train_data))
    print(someData)
    neronchik = Neuron(weights, someData, 0.01, new_train_data[0:300])
    neronchik.train_neuron()
    print(neronchik.solve_task(0))
    print(someData[0])


if __name__ == "__main__":
    main()