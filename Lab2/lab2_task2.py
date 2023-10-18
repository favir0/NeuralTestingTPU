import os
import pickle
import random
import time

import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

from neuron import Neuron


# Считывание данных
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = {}
    data['images'] = np.array(X_train, dtype='float32').tolist()
    data['labels'] = np.array(y_train, dtype='float32').tolist()

    testData = {}
    testData['images'] = np.array(X_test, dtype='float32').tolist()
    testData['labels'] = np.array(y_test, dtype='float32').tolist()

    return data, testData
        

def get_one_number(number_label, data):
    for i in range (len(data['labels'])):
        if (number_label == data['labels'][i]):
            return data['images'][i]


def get_answers_list(chosen_number):
    answers = []
    for i in range(10):
        if i == chosen_number:
            answers.append(1)
            continue
        answers.append(0)
    return answers


def get_random_test_number(test_data):
    randIndex = random.randint(0, (len(test_data['labels']) - 1))
    return (test_data['images'][randIndex], test_data['labels'][randIndex])


def flattenNumberList(numberList):
        temp = []
        [[[1, 2, 3 ,4]]]
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
        while True:
            #input("Введите любое значение, чтобы проверить случайную цифру тестовой выборки")
            totalCount += 1
            image, image_label = get_random_test_number(test_data)
            #plt.imshow(image, cmap="gray")
            #plt.show()
            #time.sleep(1)
            #plt.close('all')
            flattenImage = flattenNumberList(image)
            print(
                f"Got random image of {image_label}\n"
                f"Trying to identify is given random image a {chosen_number} or not..."
            )
            neuroResult = neuron.solve_specific_task(flattenImage)
            if image_label == chosen_number and neuroResult == 1:
                print("Success, random image is a ", chosen_number, f"| {chosen_number} == {image_label}")
                successCount += 1
            elif image_label != chosen_number and neuroResult == 0:
                print("Success, random image is not a", chosen_number, f"| {chosen_number} != {image_label}")
                successCount += 1
            else:
                print("Something went wrong...")
                print("Chosen number for neuron is ", chosen_number)
                print("Random number is ", image_label)
                print("Neruon result is", neuroResult)
            successRate = successCount / totalCount * 100
            print( "------------------------------------")
            print(f"| Total numbers checked = {totalCount} |")
            print(f"| Successfully checked = {successCount} |")
            print(f"| Success rate = {successRate}% |")
            print( "------------------------------------")


def train_neuron_for_digit(chosen_number, flatten_train_data, lr, train_data):
    answers = []
    print(f"Neuron {chosen_number}")
    for label in train_data["labels"][0:len(flatten_train_data)]:
        if (label == chosen_number):
            answers.append(1)
        else:
            answers.append(0)
    
    weights = [random.random() for _ in flatten_train_data[0]]
    neuron = Neuron(weights, answers, lr, flatten_train_data)
    neuron.train_neuron()
    return chosen_number, neuron


def save_weights(results_data):
    print("Сохранение весов нейронов в pickle файл...")
    saved_weights = {}
    
    saved_weights = dict(zip(list(map(lambda x: x[0], results_data)),list(map(lambda x: x[1].weights, results_data))))

    print(len(saved_weights[0]))
    with open('dump.pkl', 'wb') as dump_out:
        pickle.dump(saved_weights, dump_out)
    print("Веса успешно сохранены в файл dump.pkl")

def load_weights():
    if not os.path.exists('dump.pkl'):
        print("Файл dump.pkl не найден")
        return
    with open('dump.pkl', 'rb') as dump_in:
        res = pickle.load(dump_in)
        print("Веса для нейронов успешно загружены")
        return res
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
        return
    
    answer = input("Провести загрузку сохранённых весов из файла? (Y/N)")
    if (answer == "Y" or answer == "y" or answer == "Yes" or answer == "yes"):
        saved_weights_dict = load_weights()
        neurons = []
        print(f"Тестирование нейронов для всей тестовой выборки из {len(test_data['labels'])}")
        for i in range (10):
            neurons.append(Neuron([],[],1, []))
            neurons[i].load_weights(saved_weights_dict[i])
            success_rate = round(check_full_test(neurons[i], i, test_data)*100, 2)
            print(f"Нейрон {i}: {success_rate}% ")
        return
        
        
    flatten_train_images = []
    lr = 0.01
    amount = 1000
    for i, image in enumerate(train_data["images"][0:amount]):
        flatten_train_images.append(flattenNumberList(image))
    
    print(f"Нейроны будут обучены на выборке из {amount} тренировочных изображений при lr = {lr}")
    results = []
    for i in range (10):
        results.append(train_neuron_for_digit(i, flatten_train_images, lr, train_data))

    save_weights(results)
            
if __name__ == "__main__":
    main()