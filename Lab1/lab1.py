from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from numba import jit
from tqdm import tqdm
import math

# Считывание данных
def LoadData():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = {}
    data['images'] = np.array(X_train, dtype='float32')
    data['labels'] = np.array(y_train, dtype='float32')

    testData = {}
    testData['images'] = np.array(X_test, dtype='float32')
    testData['labels'] = np.array(y_test, dtype='float32')

    print("Количество изображений в обучающей выборке: ", len(X_train))
    return data, testData


# Алгоритм ближайшего соседа
def Main():
    trainData, testData = LoadData()
    
    rightGuessCount = 0
    testTotalCount = testData['images'].shape[0]

    for testIndex, testNumber in enumerate(pbar := tqdm(testData['images'])):
        rightIndex = 0
        minDiff = -1
        for trainIndex, trainNumber in enumerate(trainData['images']):
            diff = math.sqrt(((testNumber - trainNumber) ** 2).sum())
            if (minDiff < 0 or diff < minDiff):
                minDiff = diff
                rightIndex = trainIndex
        if trainData['labels'][rightIndex] == testData['labels'][testIndex]:
            rightGuessCount +=1
        pbar.set_description(f"{testIndex} Right results: {rightGuessCount}/{testTotalCount} ({rightGuessCount/(testIndex+1)*100}%)")
            
if __name__ == "__main__":
    Main()
