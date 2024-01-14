import numpy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from keras.datasets import mnist

# The following code is used for hiding the warnings and make this notebook clearer.
import warnings
warnings.filterwarnings('ignore')

def threshold(x):
  return (x >= 1).astype(int)

def threshold_der(x):
    return 1

def tanh(x):
    return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))

def tanh_derivative(x):
    return (1 + x)*(1 - x)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + numpy.exp(-x))

def sigmoid_der(x):
    """First derivative of sigmoid activation function"""
    return numpy.multiply(sigmoid(x), 1-sigmoid(x))


class NeuralNetwork:
    #########
    # parameters
    # ----------
    # self:      the class object itself
    # net_arch:  consists of a list of integers, indicating
    #            the number of neurons in each layer, i.e. the network architecture
    #########
    def __init__(self, net_arch):
        numpy.random.seed(0)
        
        # Initialized the weights, making sure we also 
        # initialize the weights for the biases that we will add later
        self.activity = threshold
        self.activity_derivative = threshold_der
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.weights = []

        # Random initialization with range of weight values (-1,1)
        for layer in range(self.layers - 1):
            w = (2*numpy.random.rand(net_arch[layer], net_arch[layer+1]) - 1)
            self.weights.append(w)
    
    def _forward_prop(self, x):
        y = x
        for i in range(len(self.weights)-1):
            activation = numpy.dot(y[i], self.weights[i])
            activity = self.activity(activation)

            # add the bias for the next layer
            y.append(activity)

        # last layer
        activation = numpy.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)
        return y
    
    def _back_prop(self, y, target, learning_rate):
        error = target - y[-1]
        delta_vec = [error * self.activity_derivative(y[-1])]

        # we need to begin from the back, from the next to last layer
        for i in range(self.layers-2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i].T)
            error = error*self.activity_derivative(y[i])
            delta_vec.append(error)

        # Now we need to set the values from back to front
        delta_vec.reverse()
        
        # Finally, we adjust the weights, using the backpropagation rules
        for i in range(len(self.weights)):
            #print(i)
            layer = y[i].reshape(1, self.arch[i])
            delta = delta_vec[i].reshape(1, self.arch[i+1])
            #print(layer.T.dot(delta))
            self.weights[i] += learning_rate*layer.T.dot(delta)

            
    
    #########
    # parameters
    # ----------
    # self:    the class object itself
    # data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
    # labels:  the result of the logical operation 'xor' on each of those input pairs
    #########
    def fit(self, data, labels, learning_rate=0.1, epochs=100):
        #print(self.weights)
        # Add bias units to the input layer - 
        # add a "1" to the input data (the always-on bias neuron)
        for k in range(epochs):
            if (k+1) % 10000 == 0:
                print('epochs: {}'.format(k+1))
        
            for i in range(len(data)):
                # We will now go ahead and set up our feed-forward propagation:
                x = [data[i]]
                y = self._forward_prop(x)
                #print(data[i])

                # Now we do our back-propagation of the error to adjust the weights:
                target = labels[i]
                self._back_prop(y, target, learning_rate)
    
    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    # 
    # parameters
    # ----------
    # self:   the class object itself
    # x:      single input data
    #########
    def predict_single_data(self, x):
        val = x
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
        return val
    
    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    # 
    # parameters
    # ----------
    # self:   the class object itself
    # X:      the input data array
    #########
    def predict(self, X):
        Y = numpy.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            Y = numpy.vstack((Y,y))
        print(Y)
        return Y
    numpy.random.seed(0)

# Initialize the NeuralNetwork with
# 2 input neurons
# 2 hidden neurons
# 1 output neuron
nn = NeuralNetwork([2,3,2,1])

# Set the input data
X = numpy.array([[0, 0], [0, 1],
                [1, 0], [1, 1]])

# Set the labels, the correct results for the xor operation
y = numpy.array([0, 1, 
                 1, 0])

# Call the fit function and train the network for a chosen number of epochs
nn.fit(X, y, epochs=10000)

# Show the prediction results
print("Final prediction")
for s in X:
    print(s, nn.predict_single_data(s))