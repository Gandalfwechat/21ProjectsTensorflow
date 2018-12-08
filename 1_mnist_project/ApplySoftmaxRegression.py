import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def DemoSoftmaxRegression(iterTimes):
    mnistData = input_data.read_data_sets('./mnist_data',one_hot=True)

    # define the regression model:
    x  = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    yy= tf.nn.softmax(tf.matmul(x,W)+b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(yy)))
    trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # define the model accuacy calculation:
    accuacyTrueFalse = tf.equal(tf.argmax(y_,1), tf.argmax(yy,1))
    accuacy = tf.reduce_mean(tf.cast(accuacyTrueFalse,tf.float32))

    batchXTest = mnistData.test.images
    batchYTest = mnistData.test.labels

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(iterTimes): 
            batchX, batchY = mnistData.train.next_batch(100)
            sess.run(trainStep,feed_dict={x: batchX, y_:batchY})
            testResults = sess.run(accuacy, feed_dict={x: batchXTest, y_:batchYTest})
            print("Step: "+str(i)+' '+str(testResults))


# convolution net  
mnistData = input_data.read_data_sets('./mnist_data',one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])


x_img = tf.reshape(x,[-1,28,28,1])
             #    batch size, len size, wid size, chennal size; 
def bias_variables(shapeUse): 
    return tf.Variable(tf.constant(0.1, shape=shapeUse))

def weight_variables(shapeUse):
    initial = tf.truncated_normal(shapeUse,stddev = 0.1)
    return tf.Variable(initial)


def con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding ="SAME")
    








#if __name__ =="__main__":

#DemoSoftmaxRegression(200)





