import numpy as np


class Network:
    def __init__(self):
        self.NeuronsInLayer = 200
        self.Layers = 5 # must be greater than 2 - we always want to have at least 1 hidden layer
        self.NeuronsByLayer = 10
        self.a = []
        self.weights = []
        self.biases = [] 

        self.initialiseData()
        
        self.weights.append(np.random.randn(self.x.shape[1], self.NeuronsByLayer))
        for i in range(self.Layers-2):
            self.weights.append(np.random.randn(self.NeuronsByLayer, self.NeuronsByLayer))
            self.biases.append(np.random.randn(self.NeuronsByLayer))

        self.weights.append(np.random.randn(self.NeuronsByLayer, self.y.shape[1]))
        self.biases.append(np.random.randn(self.y.shape[1]))

    def initialiseData(self):
        my_data = np.genfromtxt('c:/Users/Mateusz/Desktop/SN_Proj1/classification/data.simple.train.100.csv', delimiter=',')

        self.x = my_data[1:,:2]
        self.y = my_data[1:,2:3]

    def trainNetwork(self):
        for x,y in zip(self.x,self.y):
            activation_list = [self.activationFunction(x)]
            z_list = []
            for l in range(self.Layers-1): # hidden layers
                z = np.dot(activation_list[-1], self.weights[l])+self.biases[l]
                activation_list.append(self.activationFunction(z))

                z_list.append(z)
            # get delta for output
            biases_d = [np.zeros(b.shape) for b in self.biases]
            weight_d = [np.zeros(w.shape) for w in self.weights]

            # output error
            delta = np.multiply(activation_list[-1]-y, self.activateFunctionDeriv(z_list[-1]))
            biases_d[-1] = delta
            weight_d[-1] = np.dot(delta, activation_list[-2].transpose())
            
            #backpropagate
            for l in range(2, self.Layers):
                delta = np.multiply(np.dot(self.weights[1-l].transpose(), delta), self.activateFunctionDeriv(z_list[-l]))

                biases_d[-l] = delta
                weight_d[-l] = np.dot(delta, activation_list[1-l].transpose())
            print(biases_d)
            exit(0)



        


    "todo - more functions"
    @staticmethod
    def activationFunction(z): 
        return (1/(1+np.exp(-z)))

    @staticmethod
    def activateFunctionDeriv(z):
        return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))

    def runNetwork(self, x, y):
        c = x
        return c

"""
start
"""

myNetwork = Network()
myNetwork.trainNetwork()