import numpy as np


def norm(arr1):
    return np.sqrt(sum(arr1**2))

def dist(arr1, arr2):
    return norm(arr1 - arr2)

def batchGD(f, gradf, data, label):
    sampleSize, dataDim = np.shape(data)
    w = np.random.normal(size=dataDim)
    maxIter, tol, learningRate, iterationNum, err = 100000, 0.0005, 0.01, 0, np.inf
    while err > tol and iterationNum < maxIter:
        gradErr = 1.0/sampleSize * sum([(f(data[i], w)  - 2 * label[i]) * gradf(data[i], w) for i in range(sampleSize)])
        w = w - learningRate * gradErr
        err = np.sqrt(1.0/sampleSize * sum([(label[i] - f(data[i], w))**2]))
        iterationNum = iterationNum + 1
    return w

if __name__ == '__main__':
    f = lambda x, w : w[2] * x[2]**2 + w[1] * x[1] + w[0] * x[0]
    gradf = lambda x, w : np.array([x[0], x[1], x[2]**2])
    sampleSize = 200
    w = np.array([3, 7, 1])     
    x = np.random.rand(sampleSize, 3)
    label = np.array([f(x[i], w) for i in range(sampleSize)])
    wpred = batchGD(f, gradf, x, label)
    print wpred
    
    
