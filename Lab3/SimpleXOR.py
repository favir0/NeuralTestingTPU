import numpy as np
from keras.datasets import mnist

# Считывание данных
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data = {}
    data['images'] = np.array(X_train, dtype='float32')
    data['labels'] = np.array(y_train, dtype='float32')

    testData = {}
    testData['images'] = np.array(X_test, dtype='float32')
    testData['labels'] = np.array(y_test, dtype='float32')

    print("Количество изображений в обучающей выборке: ", len(X_train))
    return data, testData

# Пороговая функция 
def activation(x):
    return 0 if x <= 0 else 1

# Прямое распространение по двум нейронам 1 слой
def forward_with_com(C):
    x = np.array([C[0], C[1], 1])
    # Вручную задаём веса и смещение
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1,w2])
    w_out = np.array([-1, 1, -0.5])
    print("w_hidden: \n", w_hidden)
    print("w_out: \n", w_out)
    print("x: \n", x)
    sum = np.dot(w_hidden, x)
    print("dot_sum (w_hidden & x): \n", sum)
    out = [activation(x) for x in sum]
    out.append(1)
    out = np.array(out)
    print("out: \n", out)
    sum = np.dot(w_out, out)
    print("result sum: \n", sum)
    y = activation(sum)
    print("\n\n\n")
    return y

def forward(C):
    x = np.array([C[0], C[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1,w2])
    w_out = np.array([-1, 1, -0.5])
    sum = np.dot(w_hidden, x)
    out = [activation(x) for x in sum]
    out.append(1)
    out = np.array(out)
    sum = np.dot(w_out, out)
    y = activation(sum)
    return y

def main():
    C1 = [(1,0), (0,1)]
    C2 = [(0,0), (1,1)]
    forward_with_com(C1[0])
    print(forward(C1[0]), forward(C1[1]))
    print(forward(C2[0]), forward(C2[1]))

if __name__ == "__main__":
    main()