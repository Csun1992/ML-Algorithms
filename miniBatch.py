import numpy as np
from random import shuffle
from sys import exit

def getBatches(data, label, batchSize):
    sampleSize = len(label)
    shuffledIndex = range(sampleSize)
    shuffle(shuffledIndex) 
    dataShuffled = data[shuffledIndex]
    labelShuffled = label[shuffledIndex]
    batches = [(dataShuffled[i:i+sampleSize], labelShuffled[i:i+sampleSize]) for i in range(0, \
            sampleSize, batchSize)]
    return batches

def miniBatch(f, gradf, data, label):
    sampleSize, dataDim = np.shape(data)
    batchSize = 5
    batches = getBatches(data, label, sampleSize / batchSize) # make 5 batches
    maxIter, tol, learningRate, iterationNum, err = 10000, 0.0005, 0.01, 0, np.inf
    w = np.random.normal(size = dataDim)
    while err > tol and iterationNum < maxIter:
        err = 0
        for combo in batches:
            data, label = combo[0], combo[1]
            gradErr = 1.0/len(label) * sum([(f(data[i], w)  - 2 * label[i]) * gradf(data[i], w) for i in range(len(label))])
            w = w - learningRate * gradErr
            err = err + np.sqrt(1.0/sampleSize * sum([(label[i] - f(data[i], w))**2 for i in range(len(label))]))
        iterationNum = iterationNum + 1
    return w, iterationNum, err



if __name__ == '__main__':
    f = lambda x, w : w[2] * x[2]**2 + w[1] * x[1] + w[0] * x[0]
    gradf = lambda x, w : np.array([x[0], x[1], x[2]**2])
    sampleSize = 200
    w = np.array([3, 7, 1])     
    x = np.random.rand(sampleSize, 3)
    label = np.array([f(x[i], w) for i in range(sampleSize)])
    xperturbed = x + np.random.normal(size=(sampleSize, 3))
    wpred = miniBatch(f, gradf, xperturbed, label)
    print wpred
    
    
