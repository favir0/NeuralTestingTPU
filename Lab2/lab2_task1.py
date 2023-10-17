from tqdm import tqdm
import numpy as np
import math
from abc import ABC, abstractmethod

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
        max_point = 100
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



def main():
    startWeights = [0.1, 0.1]
    inp = [[0,0],
           [0,1],
           [1,0],
           [1,1]]
    answersAND = [0,0,0,1]
    answersOR = [0,1,1,1]
    learning_rate = 0.1

    print("Train neuron to solve AND operation: ")
    neuronAND = Neuron(startWeights,answersAND,learning_rate, inp)
    neuronAND.train_neuron()
    print("Neuron weights: ", neuronAND.weights)
    print("X1  ", "X2  ", "Y")
    for i in range(len(inp)):
        print(inp[i][0], "  ",inp[i][1], "  ",neuronAND.solve_task(i))

    print("Train neuron to solve OR operation: ")
    neuronOR = Neuron(startWeights,answersOR,learning_rate, inp)
    neuronOR.train_neuron()
    print("Neuron weights: ", neuronOR.weights)
    print("X1  ", "X2  ", "Y")
    for i in range(len(inp)):
        print(inp[i][0], "  ", inp[i][1], "  ",neuronOR.solve_task(i))
            
if __name__ == "__main__":
    main()
