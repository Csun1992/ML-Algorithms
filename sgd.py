import numpy as np
import matplotlib.pyplot as plt

def sdg(f, gradf, data, label):
    errHist = []
    sampleSize, dataDim = np.shape(data)
    w = np.random.normal(size=dataDim)
    maxIter, tol, learningRate, iterationNum, err = 100000, 30, 0.00001, 0, np.inf
    while err > tol and iterationNum < maxIter:
        err = 0
        for i in range(sampleSize):
            picked = np.random.randint(sampleSize, size=1)[0]
            gradErr = 1.0 / sampleSize * (f(data[picked], w)  - 2 * label[picked]) * \
            gradf(data[picked], w)
            w = w - learningRate * gradErr
            err = err + np.sqrt(1.0/sampleSize * (label[picked] - f(data[picked], w))**2)
            iterationNum = iterationNum + 1
        errHist.append(err)
    return w, errHist
        

if __name__ == '__main__':
    f = lambda x, w : w[2] * x[2]**2 + w[1] * x[1] + w[0] * x[0]
    gradf = lambda x, w : np.array([x[0], x[1], x[2]**2])
    sampleSize = 200
    w = np.array([3, 7, 1])     
    x = np.random.rand(sampleSize, 3)
    label = np.array([f(x[i], w) for i in range(sampleSize)])
    xperturbed = x + np.random.normal(size=(sampleSize, 3))
    wpred, errHist = sdg(f, gradf, xperturbed, label)
    plt.plot(errHist) 
    plt.show() 
