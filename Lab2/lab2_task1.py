import math
import random
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
from neuron import Neuron


def main():
    startWeights = [random.random(), random.random()]
    inp = [[0,0],
           [0,1],
           [1,0],
           [1,1]]
    answersAND = [0,0,0,1]
    answersOR = [0,1,1,1]
    learning_rate = 0.1

    print("Тренировка нейрона для решения операции AND: ")
    neuronAND = Neuron(startWeights,answersAND,learning_rate, inp)
    neuronAND.train_neuron()
    print("Веса нейрона: ", neuronAND.weights)
    print("X1  ", "X2  ", "Y")
    for i in range(len(inp)):
        print(inp[i][0], "  ", inp[i][1], "  ", neuronAND.solve_task(i))

    print("Тренировка нейрона для решения операции OR: ")
    neuronOR = Neuron(startWeights,answersOR,learning_rate, inp)
    neuronOR.train_neuron()
    print("Веса нейрона: ", neuronOR.weights)
    print("X1  ", "X2  ", "Y")
    for i in range(len(inp)):
        print(inp[i][0], "  ", inp[i][1], "  ", neuronOR.solve_task(i))
            
if __name__ == "__main__":
    main()
