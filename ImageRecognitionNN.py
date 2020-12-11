import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from progress.bar import IncrementalBar

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class NeuralNetwork():

    width = 40
    epochs = 2
    numUsing = 60000
    bs = 250
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    learningRate = 1e-3
    Lambda = 0

    def __init__(self, sizes, optimizer, hiddenActivation, outputActivation): 
        self.dimensions = sizes

        self.optimizer = optimizer
        self.hiddenActivation = hiddenActivation
        self.outputActivation = outputActivation

        self.x = np.arange(1,((self.numUsing/self.bs)*self.epochs)+1)
        self.y = np.empty(int(((self.numUsing/self.bs)*self.epochs)))

        self.secondLayerNeurons = np.empty(sizes[1])
        self.outputNeurons = np.empty(sizes[2])

        self.w1 = np.random.rand(sizes[1], sizes[0]) * 2 - 1
        self.w2 = np.random.rand(sizes[2], sizes[1]) * 2 - 1
        # self.b1 = np.random.rand(sizes[1]) * 2 - 1
        # self.b2 = np.random.rand(sizes[2]) * 2 - 1
        self.b1 = np.zeros([sizes[1]])
        self.b2 = np.zeros([sizes[2]])

        self.dw1 = np.zeros([sizes[1], sizes[0]])
        self.dw2 = np.zeros([sizes[2], sizes[1]])
        self.db1 = np.zeros([sizes[1]])
        self.db2 = np.zeros([sizes[2]])

        self.V_dw1 = np.zeros([sizes[1], sizes[0]])
        self.V_dw2 = np.zeros([sizes[2], sizes[1]])
        self.V_db1 = np.zeros([sizes[1]])
        self.V_db2 = np.zeros([sizes[2]])

        self.S_dw1 = np.zeros([sizes[1], sizes[0]])
        self.S_dw2 = np.zeros([sizes[2], sizes[1]])
        self.S_db1 = np.zeros([sizes[1]])
        self.S_db2 = np.zeros([sizes[2]])

        self.V_dw1_correct = np.empty([sizes[1], sizes[0]])
        self.V_dw2_correct = np.empty([sizes[2], sizes[1]])
        self.V_db1_correct = np.empty(sizes[1])
        self.V_db2_correct = np.empty(sizes[2])

        self.S_dw1_correct = np.empty([sizes[1], sizes[0]])
        self.S_dw2_correct = np.empty([sizes[2], sizes[1]])
        self.S_db1_correct = np.empty(sizes[1])
        self.S_db2_correct = np.empty(sizes[2])

        self.hiddenLayerErrors = np.empty(sizes[1])
        self.outputLayerErrors = np.empty(sizes[2])

    def sigmoid(self, x):
        warnings.filterwarnings("ignore")
        return 1/(1+np.exp(-x))

    def sigmoidDerivative(self, x):
        return np.multiply(x,(1-x))

    def relu(self, x):
	    return np.maximum(0.0, x)
    
    def reluDerivative(self, x):
        return 1 * (x > 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def forwardProp(self, inputs):
        if self.hiddenActivation == 'sigmoid':
            self.secondLayerNeurons = self.sigmoid(self.w1 @ inputs + self.b1)
        elif self.hiddenActivation == 'relu':
            self.secondLayerNeurons = self.relu(self.w1 @ inputs + self.b1)

        if self.outputActivation == 'sigmoid':
            self.outputNeurons = self.sigmoid(self.w2 @ self.secondLayerNeurons + self.b2)
        elif self.outputActivation == 'softmax':
            self.outputNeurons = self.softmax(self.w2 @ self.secondLayerNeurons + self.b2)

    def backProp(self, inputs, correct_output):
        self.outputLayerErrors = np.subtract(self.outputNeurons, correct_output)
        if self.hiddenActivation == 'sigmoid':
            self.hiddenLayerErrors = np.multiply(np.dot(self.w2.T, self.outputLayerErrors), self.sigmoidDerivative(self.secondLayerNeurons))
        elif self.hiddenActivation == 'relu':
            self.hiddenLayerErrors = np.multiply(np.dot(self.w2.T, self.outputLayerErrors), self.reluDerivative(self.secondLayerNeurons))

        self.db2 += self.outputLayerErrors
        self.dw2 += np.outer(self.outputLayerErrors, self.secondLayerNeurons)

        self.db1 += self.hiddenLayerErrors
        self.dw1 += np.outer(self.hiddenLayerErrors, inputs)

    def change(self):
        if self.optimizer == 'vanilla':
            self.b2 -= self.learningRate * self.db2
            self.w2 -= self.learningRate * (self.dw2 + self.Lambda * self.w2)
            self.b1 -= self.learningRate * self.db1
            self.w1 -= self.learningRate * (self.dw1 + self.Lambda * self.w1)

        elif self.optimizer == 'momentum':
            self.V_db2 = self.beta1*self.V_db2 + (1-self.beta1)*self.db2
            self.V_dw2 = self.beta1*self.V_dw2 + (1-self.beta1)*self.dw2
            self.V_db1 = self.beta1*self.V_db1 + (1-self.beta1)*self.db1
            self.V_dw1 = self.beta1*self.V_dw1 + (1-self.beta1)*self.dw1

            self.b2 -= self.learningRate * self.V_db2
            self.w2 -= self.learningRate * (self.V_dw2 + self.Lambda * self.w2)
            self.b1 -= self.learningRate * self.V_db1
            self.w1 -= self.learningRate * (self.V_dw1 + self.Lambda * self.w1)
        
        elif self.optimizer == 'adam':
            self.V_db2 = self.beta1*self.V_db2 + (1-self.beta1)*self.db2
            self.V_dw2 = self.beta1*self.V_dw2 + (1-self.beta1)*self.dw2
            self.V_db1 = self.beta1*self.V_db1 + (1-self.beta1)*self.db1
            self.V_dw1 = self.beta1*self.V_dw1 + (1-self.beta1)*self.dw1

            self.S_db2 = self.beta2*self.S_db2 + (1-self.beta2)*np.square(self.db2)
            self.S_dw2 = self.beta2*self.S_dw2 + (1-self.beta2)*np.square(self.dw2)
            self.S_db1 = self.beta2*self.S_db1 + (1-self.beta2)*np.square(self.db1)
            self.S_dw1 = self.beta2*self.S_dw1 + (1-self.beta2)*np.square(self.dw1)


            self.V_db2_correct = self.V_db2/(1-np.power(self.beta1, self.bs))
            self.V_dw2_correct = self.V_dw2/(1-np.power(self.beta1, self.bs))
            self.V_db1_correct = self.V_db1/(1-np.power(self.beta1, self.bs))
            self.V_dw1_correct = self.V_dw1/(1-np.power(self.beta1, self.bs))

            self.S_db2_correct = self.S_db2/(1-np.power(self.beta2, self.bs))
            self.S_dw2_correct = self.S_dw2/(1-np.power(self.beta2, self.bs))
            self.S_db1_correct = self.S_db1/(1-np.power(self.beta2, self.bs))
            self.S_dw1_correct = self.S_dw1/(1-np.power(self.beta2, self.bs))


            self.b2 -= self.learningRate * (self.V_db2_correct/(np.sqrt(self.S_db2_correct)+self.epsilon))
            self.w2 -= self.learningRate * (self.V_dw2_correct/(np.sqrt(self.S_dw2_correct)+self.epsilon))
            self.b1 -= self.learningRate * (self.V_db1_correct/(np.sqrt(self.S_db1_correct)+self.epsilon))
            self.w1 -= self.learningRate * (self.V_dw1_correct/(np.sqrt(self.S_dw1_correct)+self.epsilon))

        self.dw1 = np.zeros([self.dimensions[1], self.dimensions[0]])
        self.dw2 = np.zeros([self.dimensions[2], self.dimensions[1]])
        self.db1 = np.zeros(self.dimensions[1])
        self.db2 = np.zeros(self.dimensions[2])

    def train(self, trainImages, trainLabels):
        size = str(self.bs)
        accuracy = 0
        err_sum = 0.0
        avg_err = 0.0
        correct = 0

        batch_start_time = time.time()
        for m in range (self.bs):
            correct_output = np.zeros([self.dimensions[2]])
            correct_output[trainLabels[m]] = 1.0

            self.forwardProp(trainImages[m].flatten())
            self.backProp(trainImages[m].flatten(), correct_output)

            if np.argmax(self.outputNeurons) == int(trainLabels[m]):
                correct+=1

            error = np.amax(np.absolute(self.outputLayerErrors))
            err_sum += error
            avg_err = err_sum / (m+1)
            left = self.width * int(m/(self.bs/100)) // 100
            accuracy = str(int((correct/(m+1))*100)) + '%'
            print (str(m) + '/' + str(self.bs) + ' ['+'='*left+">"+'.'*((self.width - left)-1)+']' + " - Accuracy: " + accuracy + " - Loss: " + str(round(avg_err, 10)), end="\r")
        self.change()
        batch_end_time = str(round(time.time() - batch_start_time, 2))
        print (str(self.bs) + "/" + str(self.bs) + " ["+"="*self.width+"]" + " - " + batch_end_time + "s - Accuracy: " + accuracy + " - Loss: " + str(round(avg_err, 10)))
        return avg_err

    def predict(self, testImage):
        self.forwardProp(testImage)
        return np.argmax(self.outputNeurons)

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images/255

    nn = NeuralNetwork([784, 512, 10], 'adam', 'relu', 'softmax')

    start_time = time.time()
    for i in range (nn.epochs):
        start_time2 = time.time()
        for j in range(int(nn.numUsing/nn.bs)):
            print("Epoch", str(i+1) + "/" + str(nn.epochs) + ": Batch " + str(j+1) + "/" + str(int(nn.numUsing/nn.bs)))
            nn.y[j+i*(int(nn.numUsing/nn.bs))] = nn.train(train_images[int(j * nn.bs):int(j * nn.bs) + nn.bs], train_labels[int(j * nn.bs):int(j * nn.bs) + nn.bs])
        time2 = round((time.time() - start_time2), 2)

        if time2 > 60:
            print ("Epoch " + str(i+1) + " took " + str(round(time2/60, 2)) + " minutes\n\n")
        else:
            print ("Epoch " + str(i+1) + " took " + str(time2) + " seconds\n\n")
    end_time = time.time() - start_time

    print("\nTotal Time Used")
    if end_time > 60:
        print("Minutes: %s\n\n\n" % round((end_time/60),2))
    else:
        print("Seconds: %s\n\n\n" % round(end_time,2))

    incorrect = 0
    size1 = test_images.shape[0]
    bar1 = IncrementalBar('TESTING:', max = size1)
    for i in range (size1):
        bar1.next()
        prediction = nn.predict(test_images[i].flatten())
        correct = int(test_labels[i])
        if correct != prediction:
            incorrect += 1
    bar1.finish()
    print ("\nTESTING DONE\n" + "Correct: " + str(size1-incorrect) + "\nIncorrect: " + str(incorrect) + "\n")    

    plt.plot(nn.x, nn.y)
    plt.ylabel('Avg Error')
    plt.xlabel('Batches')
    plt.show()

    # stats_file = open("stats_file", "a+")
    # stats_file.write("Correct: " + str(size-incorrect) + "\nIncorrect: " + str(incorrect))
    # stats_file.write("\nOptimization: " + nn.optimizer + "\nActivation: " + nn.hiddenif self.hiddenActivation)
    # stats_file.write("\nNum Using: "+str(nn.numUsing) + "\nBatch Size: "+str(nn.bs) + "\nLearning Rate: "+str(nn.learningRate) + "\nLambda: "+str(nn.Lambda) + "\n\n\n")
    # stats_file.close