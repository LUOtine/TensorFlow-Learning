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
def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#convolution layer1
W_conv1=weight_variables([5,5,1,32])
B_conv1=biases_variables([32])
x_imagine=tf.reshape(x,[-1,28,28,1])
conv1=tf.nn.relu(conv2d(x_imagine,W_conv1)+B_conv1)
pool1=max_pooling(conv1)
#convolution layer2
W_conv2=weight_variables([5,5,32,64])
B_conv2=biases_variables([64])
conv2=tf.nn.relu(conv2d(pool1,W_conv2)+B_conv2)
pool2=max_pooling(conv2)
#fully connected nn
W_fc1=weight_variables([7*7*64,1024])
B_fc1=biases_variables([1024])
pool2_7x7=tf.reshape(pool2,[-1,7*7*64])
layerfc1=tf.nn.relu(tf.matmul(pool2_7x7,W_fc1)+B_fc1)
#fully connected nn output
W_fc2=weight_variables([1024,10])
B_fc2=biases_variables([10])
prediction=tf.nn.softmax(tf.matmul(layerfc1,W_fc2)+B_fc2)
#cross entropy
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),reduction_indices=[1]))
#training
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#training
for _ in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train,feed_dict={x:batch_x,y:batch_y})
    if _ % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))