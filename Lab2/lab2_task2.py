from keras.datasets import mnist
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool

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

# Класс neuron для реализации логики нейрона
class Neuron():
    def __init__(self, weights, value_to_find, learning_rate, inp):
        self.weights = weights.copy()
        self.value_to_find = value_to_find.copy()
        self.learning_rate = learning_rate
        self.inp = inp.copy()
        self.epoch = 0
    
    def solve_specific_task(self, sideInput):
        curAnswer = 0
        for i in range(len(self.weights)):
            curAnswer += self.weights[i] * sideInput[i]
        # --------------- Пороговая функция активации ---------------
        if curAnswer >= 1: 
            curAnswer = 1
        else:
            curAnswer = 0
        # -----------------------------------------------------------
        return curAnswer 

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
    
    def load_weights(self, new_weights):
        self.weights = new_weights.copy()

    def train_neuron(self, new_input = [], new_answers = []):
        if (len(new_input) != 0 or len(new_answers) != 0):
            print("before", len(self.inp))
            self.inp = new_input.copy()
            self.value_to_find = new_answers.copy()
            print("after", len(self.inp))
        epoch = 0
        complete_train = False
        max_point = 100
        time0 = time.perf_counter()
        while not complete_train:
            allAnswersRight = True
            # to do ERROR with value to find, it still has small range
            # need to update this thing to with input
            for i, result in enumerate(self.value_to_find):
                # Текущее значение функции
                curAnswer = self.solve_task(i)
                if result != curAnswer:
                    allAnswersRight = False
                    self.recalc_weights(i, curAnswer)

            max_point -=1
            epoch+=1
            if (max_point == 0):
                print("Ошибка тренировки, бесконечный цикл")
                break
            complete_train = allAnswersRight
        print(f"Обучение нейрона завершено за {time.perf_counter() - time0} seconds")
        

def get_one_number(number_label, data):
    for i in range (len(data['labels'])):
        if (number_label == data['labels'][i]):
            return data['images'][i]


def get_answers_list(chosen_number):
    answers = []
    for i in range(10):
        if i == chosen_number:
            answers.append(1)
        else:
            answers.append(0)
    return answers

def get_random_test_number(test_data):
    randIndex = random.randint(0, (len(test_data['labels']) - 1))
    return (test_data['images'][randIndex], test_data['labels'][randIndex])

def flattenNumberList(numberList):
        temp = []
        for x in numberList:
            temp.extend(x if isinstance(x, list) else [x])
        return temp

def continue_neuron_train(neuron, chosen_number, train_data, amount = 100):
    print("Amount of numbers to be included in train session: ", amount)
    flatten_train_images = []
    new_answers = []
    for i, image in enumerate(train_data["images"][0:amount]):
        flatten_train_images.append(flattenNumberList(image))
        if (train_data["labels"][i] == chosen_number):
            new_answers.append(1)
        else:
            new_answers.append(0)
    

    print(len(flatten_train_images))
    neuron.train_neuron(flatten_train_images, new_answers)
    return neuron

def check_full_test(neuron, chosen_number, test_data):
        successRate = 0
        successCount = 0
        totalCount = len(test_data["images"])
        for (i, image) in enumerate(test_data["images"]):
            flattenImage = flattenNumberList(image)
            neuroResult = neuron.solve_specific_task(flattenImage)
            if ((test_data["labels"][i] == chosen_number and neuroResult == 1) or
                (test_data["labels"][i] != chosen_number and neuroResult == 0)):
                successCount += 1
        successRate = successCount / totalCount
        return successRate

def check_random_numbers(neuron, chosen_number,test_data):
        successRate = 0
        totalCount = 0
        successCount = 0
        while(True):
            #input("Введите любое значение, чтобы проверить случайную цифру тестовой выборки")
            totalCount += 1
            image, image_label = get_random_test_number(test_data)
            #plt.imshow(image, cmap="gray")
            #plt.show()
            #time.sleep(1)
            #plt.close('all')
            flattenImage = flattenNumberList(image)
            print("Got random image of ", image_label)
            print("Trying to identify is given random image a ", chosen_number, " or not...")
            neuroResult = neuron.solve_specific_task(flattenImage)
            if (image_label == chosen_number and neuroResult == 1):
                print("Success, random image is a ", chosen_number, f"| {chosen_number} == {image_label}")
                successCount += 1
            elif (image_label != chosen_number and neuroResult == 0):
                print ("Success, random image is not a", chosen_number, f"| {chosen_number} != {image_label}")
                successCount += 1
            else:
                print ("Something went wrong...")
                print ("Chosen number for neuron is ", chosen_number)
                print ("Random number is ", image_label)
                print ("Neruon result is", neuroResult)
            successRate = successCount / totalCount * 100
            print( "------------------------------------")
            print(f"| Total numbers checked = {totalCount} |")
            print(f"| Successfully checked = {successCount} |")
            print(f"| Success rate = {successRate}% |")
            print( "------------------------------------")


def train_neuron_for_digit(chosen_number, flatten_train_data_to_copy, lr, train_data_to_copy):
    answers = []
    print("abaras")
    flatten_train_data = flatten_train_data_to_copy.copy()
    train_data = train_data_to_copy.copy()
    for label in train_data["labels"][0:len(flatten_train_data)]:
        if (label == chosen_number):
            answers.append(1)
        else:
            answers.append(0)
    
    weights = [random.random() for _ in flatten_train_data[0]]
    neuron = Neuron(weights, answers, lr, flatten_train_data)
    neuron.train_neuron()
    return chosen_number, neuron


def main():
    train_data, test_data = load_data()
    flattenNumbers = []
    weights = []
    testing = False
    if (testing):
    # Создаём тренировочную выборку по 1 экзепляру каждой цифры, переводим двумерные списки в одномерные
        for i in range (10):
            flattenNumbers.append(flattenNumberList((get_one_number(i, train_data))))

        # Создаём случайные стартовые веса для нейрона
        for i in range (len(flattenNumbers[0])):
            weights.append(random.random())

        # Заполняем массив ответов, 1 у выбранной цифры и 0 у прочих  
        chosen_number = 2  
        answers = get_answers_list(chosen_number)
        lr = 0.01
        print(answers)
        print("Learning rate: ", lr)
        neuronchik = Neuron(weights, answers, lr, flattenNumbers)
        
        # Обучение нейрона
        neuronchik.train_neuron()

        print("Проверка правильности обучения на исходных данных: ")
        for i in range (10):
            print ("Checking number ", i)
            if (answers[i] == neuronchik.solve_task(i)):
                print("Success")
            else:
                print("Error")
        continue_neuron_train(neuronchik, chosen_number, train_data, amount=1000)
        #check_random_numbers(neuronchik, chosen_number, test_data)
        print(check_full_test(neuronchik, chosen_number, test_data)*100,"%")
    else:
        flatten_train_images = []
        lr = 0.01
        amount = 1000
        for i, image in enumerate(train_data["images"][0:amount]):
            flatten_train_images.append(flattenNumberList(image))
        processes = []
        results = []
        temp_args = []
        for digit in range(10):
            temp_args.append([digit, flatten_train_images, lr, train_data])
        pool = Pool(processes=10)

        for result in pool.starmap(train_neuron_for_digit, temp_args):
                results.append(result)
        print(results)
            
            
        
    
    
if __name__ == "__main__":
    main()