import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
#define add_layer function
def add_layer(inputs,raw,column,activation_function=None):
    Weight = tf.Variable(tf.random_normal([raw,column]))
    Biases = tf.Variable(tf.zeros([1,column]))
    Wx_B = tf.add(tf.matmul(inputs,Weight),Biases)
    if activation_function is None:
        output = Wx_B
    else:
        output = activation_function(Wx_B)
    return output
#download MNIST database
mnist = input_data.read_data_sets("C:\VS-CODE-Python\MNIST_data",one_hot=True)
#define accuracy
def compute_accuracy(xs,ys):
    y_pre=sess.run(prediction,feed_dict={x:xs})
    correct=tf.equal(tf.argmax(y_pre,1),tf.argmax(ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    result=sess.run(accuracy,feed_dict={x:xs,y:ys})
    return result
#define placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#add layer
prediction=add_layer(x,784,10,activation_function=tf.nn.softmax)
#cross entropy
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),reduction_indices=[1]))
#training
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#Session part
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#training
for _ in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train,feed_dict={x:batch_x,y:batch_y})
    if _ % 500 == 0£º
        print(compute_accuracy(mnist.test.images,mnist.test.labels))