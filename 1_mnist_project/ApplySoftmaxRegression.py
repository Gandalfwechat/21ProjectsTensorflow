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

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(yy)))
    crossE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=yy))
    trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossE)

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


def bias_variables(shapeUse): 
    return tf.Variable(tf.constant(0.1, shape=shapeUse))

def weight_variables(shapeUse):
    initial = tf.truncated_normal(shapeUse,stddev = 0.1)
    return tf.Variable(initial)


def con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding ="SAME")
    
def maxPool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


def DemoMNISTbyConvolutionLayer(iterTimes):
    # convolution net  
    mnistData = input_data.read_data_sets('./mnist_data',one_hot=True)

    Xtest = mnistData.test.images
    Ytest =  mnistData.test.labels

    x = tf.placeholder(tf.float32,[None,784])
    yl = tf.placeholder(tf.float32,[None,10])

    x_img = tf.reshape(x,[-1,28,28,1])
                #    batch size, len size, wid size, chennal size; 

    #define the 1st conv layer 
    W_conv_1 = weight_variables([5,5,1,32])
    b_conv_1 = bias_variables([32]) # output 32 channel; 
    conv_re_1 = con2d(x_img,W_conv_1)
    h_conv_1 = tf.nn.relu(conv_re_1+b_conv_1) # when apply the conv_re_1+b_conv_1 
                                            # conv_re_1 become [-1, 28,28,32] 
                                            # and the b_conv_1 would be extend to [-1, 28,28,32] 
    # then define 1st max pooling layer: 
    h_pool_1= maxPool_2_2(h_conv_1)    # the pooling layer become [1,14,14,32]

    #define the 2ed conv layer 
    W_conv_2 = weight_variables([5,5,32,64])
    b_conv_2 = bias_variables([64]) # output 64 channel; 

    conv_re_2 = con2d(h_pool_1,W_conv_2)+b_conv_2
    h_conv_2 = tf.nn.relu(conv_re_2) # when apply the conv_re_2+b_conv_2 
                                            # conv_re_1 become [-1,14,14,64] 
                                            # and the b_conv_1 would be extend to [-1, 14,14,64] 
    # then define 1st max pooling layer: 
    h_pool_2 = maxPool_2_2(h_conv_2)    # the pooling layer become [-1,7,7,64]

    # then define one full connect layer 
    W_fcl = weight_variables([7*7*64,1024])
    b_fcl = bias_variables([1024])
    h_pool_2_flat = tf.reshape(h_pool_2,[-1,7*7*64])
    input_fcl = tf.matmul(h_pool_2_flat, W_fcl)+ b_fcl
    h_fcl = tf.nn.relu(input_fcl)    # dimenssion : [-1, 1024]

    # then apply keep_prob to avoid overfitting
    keep_prb = tf.placeholder(tf.float32)    
    h_fcl_drop = tf.nn.dropout(h_fcl,keep_prb)   # dimenssion: [-1, 1024]

    # defin output layer 

    w_output = weight_variables([1024,10])
    b_output = bias_variables([10])
    y_out = tf.matmul(h_fcl_drop,w_output)+ b_output 

    crossE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yl,logits=y_out))

    trainStep = tf.train.AdamOptimizer(0.001).minimize(crossE)

    correct_pre = tf.equal(tf.argmax(y_out,1),tf.argmax(yl,1))
    acc = tf.reduce_mean(tf.cast(correct_pre,tf.float32))

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        for i in range(iterTimes):
            batchX1, batchY1 = mnistData.train.next_batch(20)
            re = sess.run(acc,feed_dict={x:batchX1, yl:batchY1, keep_prb: 1.0 })
            print("Step: " + str(i) + " "+str(re))   
            sess.run(trainStep,feed_dict={x:batchX1, yl:batchY1, keep_prb: 1.0 }) 
        
        re = sess.run(acc,feed_dict={x:Xtest, yl:Ytest, keep_prb: 1.0 })
        print( "final accuracy: "+str(re))   


if __name__ =="__main__":

    DemoSoftmaxRegression(200)
    DemoMNISTbyConvolutionLayer(200)




