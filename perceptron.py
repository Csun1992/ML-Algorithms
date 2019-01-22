import numpy as np
import matplotlib.pyplot as plt

# we try to implement a simple PLA for separable data
def predict(weight, data):
    bias = weight[0]
    parameterWeight = weight[1:]
    sign = lambda x : -1 if x < 0 else 1 if x > 0 else 0
    result = sign(bias + parameterWeight.dot(data))
    return result 

def train(data, result):
    sampleSize, dataDim = np.shape(data)
    allClassified = False
    misclassifiedIndex = np.random.randint(0, sampleSize) 
    weight = np.zeros(dataDim + 1)
    while not allClassified:
        weight[0] = weight[0] + result[misclassifiedIndex]
        weight[1:] = weight[1:] + result[misclassifiedIndex] * data[misclassifiedIndex]
        allClassified = True
        for i in range(sampleSize):
            if result[i] != predict(weight, data[i]):
                misclassifiedIndex = i
                allClassified = False
                break
    return weight


                
if __name__ == '__main__':
    target = np.random.rand(3)
    data = np.random.randint(-20, 20, size=(1000, 2))
    result = []
    length = np.size(data, axis=0)
    for i in range(length):
        result.append(predict(target, data[i]))
    w = train(data, result)
    prediction = []
    for i in range(length):
        prediction.append(predict(w, data[i, :]))
    print sum([x -  y for (x,y) in zip(prediction, result)])
    w0 = target[0]
    w1 = target[1]
    w2 = target[2]
    x = np.linspace(-20,20, 100)
    y = -w1/w2 * x - w0 / w2
    marker = ['o', 'x', '*']
    color = ['r', 'g', 'k']
    result = np.array(result)
    for i, c in enumerate(np.unique(result)):
        plt.scatter(data[:, 0][result==c], data[:, 1][result==c], c=color[i], marker=marker[i])
    plt.plot(x, y)
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    x = np.linspace(-20,20, 100)
    y = -w1/w2 * x - w0 / w2
    plt.plot(x, y)
    plt.show()
