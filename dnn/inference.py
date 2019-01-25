import tensorflow as tf

def getLayerName(num):
    return 'layer' + str(num)

def getWeightVariable(shape, regularizer):
    weights = tf.get_variable("weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(inputTensor, layerNum, layerNodes, outputNode):
    inputNode = inputTensor.shape[0]
    layer = inputTensor 
    for i in range(layerNum):
        with tf.variable_scope(getLayerName(i)):
            weights = get_weight_variable([layerNodes[i], layerNodes[i+1]], regularizer)
            biases = tf.get_variable("biases", [layerNodes[i+1]])
            layer = tf.matmul(layer, weights) + biases
    return layer

