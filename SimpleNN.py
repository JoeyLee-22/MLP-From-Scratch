import time
import copy
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork():

    width = 20
    correct = 0
    num_predictions = 10
    learningRate = 0.1
    beta = 0.9
    Lambda = 0
    epochs = 10000
    correctTarget = 0.95

    def __init__(self, sizes, sizeOfEpoch):
        self.sizeOfEpoch = sizeOfEpoch 
        self.dimensions = sizes

        self.x = np.arange(1,self.epochs+1)
        self.accuracyY = np.empty(self.epochs)
        self.errorY = np.empty(self.epochs)

        self.secondLayerNeurons = np.empty(sizes[1])
        self.outputNeurons = np.empty(sizes[2])

        self.firstLayerWeights = np.random.rand(sizes[1], sizes[0]) * 2 - 1
        self.secondLayerWeights = np.random.rand(sizes[2], sizes[1]) * 2 - 1
        self.firstLayerBiases = np.random.rand(sizes[1]) * 2 - 1
        self.secondLayerBiases = np.random.rand(sizes[2]) * 2 - 1

        self.firstLayerWeightsSummations = np.zeros([sizes[1], sizes[0]])
        self.secondLayerWeightsSummations = np.zeros([sizes[2], sizes[1]])
        self.firstLayerBiasesSummations = np.zeros([sizes[1]])
        self.secondLayerBiasesSummations = np.zeros([sizes[2]])

        self.firstLayerWeightsVelo = np.zeros([sizes[1], sizes[0]])
        self.secondLayerWeightsVelo = np.zeros([sizes[2], sizes[1]])
        self.firstLayerBiasesVelo = np.zeros([sizes[1]])
        self.secondLayerBiasesVelo = np.zeros([sizes[2]])

        self.hiddenLayerErrors = np.empty(sizes[1])
        self.outputLayerErrors = np.empty(sizes[2])
        self.greatestError = np.zeros(1)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerivative(self, x):
        return np.multiply(x,(1-x))

    def forwardProp(self, inputs):
        for i in range (self.dimensions[1]):
            self.secondLayerNeurons[i] = self.sigmoid(np.dot(self.firstLayerWeights[i], inputs)+self.firstLayerBiases[i])
        for i in range (self.dimensions[2]):
            self.outputNeurons[i] = self.sigmoid(np.dot(self.secondLayerWeights[i], self.secondLayerNeurons)+self.secondLayerBiases[i])

    def backProp(self, inputs, correct_output):
        self.outputLayerErrors = np.subtract(self.outputNeurons, correct_output)
        self.hiddenLayerErrors = np.multiply(np.dot(self.secondLayerWeights.T, self.outputLayerErrors), self.sigmoidDerivative(self.secondLayerNeurons))

        self.secondLayerBiasesSummations += self.outputLayerErrors
        self.secondLayerWeightsSummations += np.outer(self.outputLayerErrors, self.secondLayerNeurons)

        self.firstLayerBiasesSummations += self.hiddenLayerErrors
        self.firstLayerWeightsSummations += np.outer(self.hiddenLayerErrors, inputs)

    def change(self):
        self.secondLayerBiasesVelo = self.beta*self.secondLayerBiasesVelo + (1-self.beta)*self.secondLayerBiasesSummations
        self.secondLayerWeightsVelo = self.beta*self.secondLayerWeightsVelo + (1-self.beta)*self.secondLayerWeightsSummations
        self.firstLayerBiasesVelo = self.beta*self.firstLayerBiasesVelo + (1-self.beta)*self.firstLayerBiasesSummations
        self.firstLayerWeightsVelo = self.beta*self.firstLayerWeightsVelo + (1-self.beta)*self.firstLayerWeightsSummations

        self.secondLayerBiases -= self.learningRate * self.secondLayerBiasesVelo
        self.secondLayerWeights -= self.learningRate * (self.secondLayerWeightsVelo + self.Lambda * self.secondLayerWeights)
        self.firstLayerBiases -= self.learningRate * self.firstLayerBiasesVelo
        self.firstLayerWeights -= self.learningRate * (self.firstLayerWeightsVelo + self.Lambda * self.firstLayerWeights)

        self.firstLayerWeightsSummations = np.zeros([self.dimensions[1], self.dimensions[0]])
        self.secondLayerWeightsSummations = np.zeros([self.dimensions[2], self.dimensions[1]])
        self.firstLayerBiasesSummations = np.zeros(self.dimensions[1])
        self.secondLayerBiasesSummations = np.zeros(self.dimensions[2])

    def train(self, trainImages, trainLabels):
        accuracy = 0
        size = str(self.sizeOfEpoch)
        start_time2 = time.time()

        for m in range (self.sizeOfEpoch):
            correct_output = trainLabels[m]

            self.forwardProp(trainImages[m].flatten())
            self.backProp(trainImages[m].flatten(), correct_output)

            if np.absolute(self.outputLayerErrors) > self.greatestError:
                self.greatestError = np.absolute(self.outputLayerErrors)
            if self.outputNeurons > self.correctTarget and trainLabels[m] == 1 or self.outputNeurons < 0.1 and trainLabels[m] == 0:
                self.correct+=1
            accuracy = str(int((self.correct/(m+1))*100)) + '%'
            percent = str(int((m/self.sizeOfEpoch)*100)) + '%'
            print ("Progress: " + percent + " -- Accuracy: " + accuracy, end="\r")
        self.change()

        time2 = str(round((time.time() - start_time2), 2))
        print (size+'/'+size+" -- "+time2+"s"+" -- Accuracy: "+accuracy+" -- Error:",self.greatestError,end="\r")
        correct = np.copy(self.correct)
        self.correct = 0
        return (correct/self.sizeOfEpoch)
            
    def predict(self, testImage):
        self.forwardProp(testImage)
        return self.outputNeurons

if __name__ == "__main__":

    train_images = np.array([[1,0,0,1],
                            [1,0,1,0],
                            [0,1,0,1],
                            [0,0,0,0],
                            [1,1,1,1]])
    train_labels = np.array([1, 0, 0, 0, 0])
    # train_images = np.array([[0,0], [0,1], [1,0], [1,1]])
    # train_labels = np.array([0,1,1,0])

    neural_network = NeuralNetwork([4, 3, 1], train_images.shape[0])

    start_time = time.time()
    for i in range (neural_network.epochs):
        print ("\nEpoch", str(i+1) + "/" + str(neural_network.epochs))
        neural_network.accuracyY[i]=neural_network.train(train_images, train_labels)
        neural_network.errorY[i] = np.copy(neural_network.greatestError)
        neural_network.greatestError = 0
    time = time.time() - start_time

    plt.plot(neural_network.x, neural_network.errorY, 'r')
    plt.plot(neural_network.x, neural_network.accuracyY, 'b')
    plt.ylabel('Change In Error/Accuaracy')
    plt.xlabel('Epochs')
    plt.show()

    print("\n\n\nTotal Time Used")
    if time > 60:
        print("Minutes: %s" % round((time/60),2))
    else:
        print("Seconds: %s" % round(time,2))

    for i in range (neural_network.num_predictions):
        print("\n\nNew Situations: " + str(i+1) + "/" + str(neural_network.num_predictions))
        A = list(map(int,input("Enter the numbers : ").strip().split()))[:4] 
   
        try:
            result = neural_network.predict(A)
        except:
            print("\nError, try again")
            continue

        print("\nOutput Data:", result)
       
        if result>neural_network.correctTarget:
            print("Result: Back Slash")
        else:
            print("Result: Not Back Slash")
        # if result>0.95:
        #     print("Result: 1")
        # else:
        #     print("Result: 0")