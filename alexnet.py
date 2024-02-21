# coding: UTF-8

# based on Frederik Kratzert's alexNet with tensorflow
import tensorflow as tf
import numpy as np
# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):

    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

class alexNet(object):
    """alexNet model"""
    def __init__(self, x, keepPro, classNum, skip, modelPath = "bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        #build CNN
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    def train(self, sess, train_data, train_labels, num_epochs, batch_size, learning_rate):
        """Train the model"""
        # Define placeholders for labels
        self.Y = tf.placeholder(tf.float32, [None, self.class_num], name='Y')
        
        # Define loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        
        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Define accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Training loop
        for epoch in range(num_epochs):
            for i in range(0, len(train_data), batch_size):
                batch_x = train_data[i:i + batch_size]
                batch_y = train_labels[i:i + batch_size]
                # Run optimization op (backprop) and compute loss
                _, l = sess.run([optimizer, loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: self.keep_prob})
            # Display loss and accuracy per epoch
            acc = sess.run(accuracy, feed_dict={self.X: train_data, self.Y: train_labels, self.keep_prob: 1.0})
            print('Epoch %i: Loss: %f, Accuracy: %f' % (epoch + 1, l, acc))

    def evaluate(self, sess, test_data, test_labels):
        """Evaluate the model"""
        # Define accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Evaluate accuracy on test set
        acc = sess.run(accuracy, feed_dict={self.X: test_data, self.Y: test_labels, self.keep_prob: 1.0})
        print('Test Accuracy: %f' % acc)

    def save_model(self, sess, save_path):
        """Save the model"""
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print('Model saved to %s' % save_path)

    def load_model(self, sess, load_path):
        """Load the model"""
        saver = tf.train.Saver()
        saver.restore(sess, load_path)
        print('Model loaded from %s' % load_path)