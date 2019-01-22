import numpy as np
import matplotlib.pyplot as plt

def dist(arr1, arr2):
    return np.sqrt(sum((arr1 - arr2)**2) / float(len(arr1)))

def distFromCenter(data, centers):
    sampleSize, centerNum = np.size(data, axis = 0), np.size(centers, axis = 0)
    distances = np.array([ [dist(i, j) for i in centers] for j in data])
    return distances

def cluster(data, centers):
    dist = distFromCenter(data, centers)
    groupNum = np.argmin(dist, axis = 1) 
    return groupNum

def normalize(data):
    return (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)


def kmean(data, clusterNum):
    data = normalize(data)
    sampleSize, dataDim = np.shape(data)
    oldGroupNum = np.zeros(sampleSize)
    # randomly generate initial centers from data
    centers = data[np.random.choice(sampleSize, size=clusterNum, replace=False)] 
    finishClustering = False
    while not finishClustering:
        groupNum = cluster(data, centers)
        if (oldGroupNum == groupNum).all():
            finishClustering = True
        else: 
            oldGroupNum = groupNum
            centers = np.array([np.mean(data[groupNum == i], axis = 0) for i in range(clusterNum)]) 
    return groupNum



if __name__ == '__main__':
    a = np.random.uniform(low = -1, size=(20,2)) * 10
    print kmean(a, 4)
