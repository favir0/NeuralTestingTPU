import time

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
        return 1 if curAnswer >= 1 else 0
        # -----------------------------------------------------------

    def solve_task(self, inpIndex):
        # Текущее значение функции
        curAnswer = 0
        for i in range(len(self.weights)):
            curAnswer += self.weights[i] * self.inp[inpIndex][i]
    
        # --------------- Пороговая функция активации ---------------
        return 1 if curAnswer >= 1 else 0
        # -----------------------------------------------------------

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

        # Дополнительное обучение при необходимости
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
        print(f"Обучение нейрона завершено за {time.perf_counter() - time0} секунд, количество эпох: {epoch}")