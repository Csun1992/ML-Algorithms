import tensorflow as tf

def getLayerName(num):
    return 'layer' + str(num)

def getWeightVariable(shape, regularizer):
    weights = tf.get_variable("weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# layerNums is the total number of layers INCLUDING input and output layer 
# nodesPerLayer is the list of number of nodes in each layer including the output layer
def forwardPropagation(inputTensor, layerNums, nodesPerLayer, regularizer):
    layer = inputTensor 
    for i in range(1, layerNums):
        with tf.variable_scope(getLayerName(i)):
            weights = getWeightVariable([nodesPerLayer[i-1], nodesPerLayer[i]], regularizer)
            biases = tf.get_variable("biases", [nodesPerLayer[i]])
            layer = tf.matmul(layer, weights) + biases
    return layer
