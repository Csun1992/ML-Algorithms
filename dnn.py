import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as inputData

inputNode = 784
outputNode = 10
layer1Node = 500
batchSize = 100
learningRate = 0.5
regularizationRate = 0.0001
trainingStep = 30000

def forwardPropagation(inputTensor, weights1, bias1, weights2, bias2):
    layer1 = tf.nn.relu(tf.matmul(inputTensor, weights1) + bias1)
    return tf.matmul(layer1, weights2) + bias2

def train(mnist):
    x = tf.placeholder(tf.float32, [None, inputNode], name = 'x')
    y = tf.placeholder(tf.float32, [None, outputNode], name = 'y')
    weights1 = tf.Variable(tf.truncated_normal([inputNode, layer1Node], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[layer1Node]))
    weights2 = tf.Variable(tf.truncated_normal([layer1Node, outputNode]))
    bias2 = tf.Variable(tf.constant(0.1, shape=[outputNode]))

    # now define the error in the deep learning algorithms
    prediction = forwardPropagation(x, weights1, bias1, weights2, bias2)
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels=tf.argmax(y, 1))
    crossEntropyMean = tf.reduce_mean(crossEntropy)
    regularizer = tf.contrib.layers.l2_regularizer(regularizationRate)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = crossEntropyMean + regularization

    # define traing operation by deciding optimizer and pass loss function to it 
    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
    correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validateFeed = {x : mnist.validation.images, y : mnist.validation.labels}
        testFeed = {x : mnist.test.images, y : mnist.test.labels}
    
        for i in range(trainingStep):
            xs, ys = mnist.train.next_batch(batchSize)
            sess.run(trainStep, feed_dict = {x : xs, y : ys})
        testAcc = sess.run(accuracy, feed_dict=testFeed) 
        valAcc = sess.run(accuracy, feed_dict=validateFeed)


mnist = inputData.read_data_sets("/tmp/data", one_hot=True)
train(mnist)
