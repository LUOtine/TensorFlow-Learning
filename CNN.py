#with hidden layer
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
#download MNIST database
mnist = input_data.read_data_sets("C:\VS-CODE-Python\MNIST_data",one_hot=True)
#define accuracy
def compute_accuracy(xs,ys):
    y_pre=sess.run(prediction,feed_dict={x:xs})
    correct=tf.equal(tf.argmax(y_pre,1),tf.argmax(ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    result=sess.run(accuracy,feed_dict={x:xs,y:ys})
    return result
#CNN
#biases and weight
def weight_variables(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def biases_variables(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#convolution and pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#convolution layer












#define placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#add layer
l2=add_layer(x,500,10,activation_function=tf.nn.relu)
prediction=add_layer(l2,784,10,activation_function=tf.nn.softmax)
#cross entropy
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),reduction_indices=[1]))
#training
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)