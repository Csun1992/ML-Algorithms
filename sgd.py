import numpy as np

def sdg(f, gradf, data, label):
    sampleSize, dataDim = np.shape(data)
    w = np.random.normal(size=dataDim)
    maxIter, tol, learningRate, iterationNum, err = 100000, 0.0005, 0.01, 0, np.inf
    while err > tol and iterationNum < maxIter:
        for i in range(sampleSize):
            picked = np.random.randint(sampleSize, size=1)[0]
            gradErr = 1.0 / sampleSize * (f(data[picked], w)  - 2 * label[picked]) * \
            gradf(data[picked], w)
            w = w - learningRate * gradErr
            err = np.sqrt(1.0/sampleSize * sum([(label[picked] - f(data[picked], w))**2]))
            iterationNum = iterationNum + 1
    return w, iterationNum
        

if __name__ == '__main__':
    f = lambda x, w : w[2] * x[2]**2 + w[1] * x[1] + w[0] * x[0]
    gradf = lambda x, w : np.array([x[0], x[1], x[2]**2])
    sampleSize = 200
    w = np.array([3, 7, 1])     
    x = np.random.rand(sampleSize, 3)
    label = np.array([f(x[i], w) for i in range(sampleSize)])
    wpred = sdg(f, gradf, x, label)
    print wpred
    
    
