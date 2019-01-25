import tensorflow as tf
from forwardPropagation import *
import os

batchSize = 100

# data should be numpy array or pandas dataframe
def train(data, label, layerNums, nodesPerLayer, regularizer=None, regularizationRate=0.0001,
        learningRateDecay=0.99, learningRateBase=0.8, batchSize=100, movingAverageDecay=0.99, \
        trainingSteps=30000, modelPath='./', modelName='model.ckpt'):
    sampleSize, dataDim = data.shape
    batchNum = sampleSize / batchSize + 1
    x = tf.placeholder(tf.float32, [None, dataDim], name="x")
    y = tf.placeholder(tf.float32, [None, nodesPerLayer[-1]], name='y') 
    regularizer = tf.contrib.layers.l2_regularizer(regularizationRate)
    prediction = forwardPropagation(x, layerNums, nodesPerLayer, regularizer)
    globalStep = tf.Variable(0, trainable=False)

    variableAverages = tf.train.ExponentialMovingAverage(movingAverageDecay, globalStep)
    variableAveragesOperation = variableAverages.apply(tf.trainable_variables())
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,\
            lables=tf.argmax(y, 1))
    crossEntropyMean = tf.reduce_mean(crossEntropy)
    loss = crossEntropyMean + tf.add_n(tf.get_collection('losses'))
    learningRate = tf.train.exponential_decay(learningRateBase, globalStep, batchNum,\
            learningRateDecay)
    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss, global_step=globalStep)
    trainOperation = tf.group(trainStep, variableAveragesOperation)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(trainingSteps):
            for j in range(batchNum):
                xs, ys = data[batchNum*batchSize : (batchNum+1)*batchSize], label[batchNum*batchSize : (batchNum+1)*batchSize]
            _, lossVal, step = sess.run([trainOperation, loss, globalStep], feed_dict={x : xs, \
                   y : ys})
            if i % 1000 == 0:
            print (step, lossVal)
