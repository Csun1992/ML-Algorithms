import numpy as np
from sys import exit

def dist(arr1, arr2):
    return np.sqrt(sum((arr1 - arr2)**2))

def knn(data, k, trainData, trainLabel):
    sampleSize = float(len(trainLabel))
    prediction = []
    for i in data:
        distances = np.array([dist(i, j) for j in trainData])
        nearNeibs = np.argsort(-distances)[:k]
        prediction.append(sum(trainLabel[nearNeibs]) / k)
    return np.array(prediction)


if __name__ == '__main__':
    trainData = np.random.uniform(low=-1, size=(100,2))
    trainLabel = np.random.randint(10, size=100) 
    data = np.random.uniform(low = -1, size=(10, 2))
    print knn(data, 3, trainData, trainLabel) 
