import multiprocessing

def train_neuron_for_digit(digit, train_data, answers, lr, flattenNumbers):
weights = [random.random() for _ in flattenNumbers[0]]
neronchik = Neuron(weights, answers, lr, flattenNumbers)
neronchik.train_neuron()
return digit, neronchik

def main():
train_data, test_data = load_data()
flattenNumbers = []

# Создаём тренировочную выборку по 1 экземпляру каждой цифры, переводим двумерные списки в одномерные
for i in range(10):
flattenNumbers.append(flattenNumberList(get_one_number(i, train_data)))

# Определите learning rate и другие параметры, которые будут использоваться для каждой цифры

processes = []
results = []

with multiprocessing.Pool(processes=10) as pool:
for digit in range(10):
answers = get_answers_list(digit)
lr = 0.01 # Установите свои значения
processes.append(pool.apply_async(train_neuron_for_digit, args=(digit, train_data, answers, lr, flattenNumbers)))

for process in processes:
digit, neronchik = process.get()
results.append((digit, neronchik))

for digit, neronchik in results:
print(f"Neuron for digit {digit} trained.")

if __name__ == "__main__":
main()