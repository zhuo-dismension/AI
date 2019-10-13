# 2019.10.13
"""
author:Tzhuo
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

DATA_DIR='/tmp/data'
NUM_STEPS=1000
MINIBATCH_SIZE=100
data=input_data.read_data_sets(DATA_DIR,one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
w=tf.Variable(tf.zeros([784,10]))

y_true=tf.placeholder(tf.float32,[None,10])
y_pred=tf.matmul(x,w)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits=y_pred,labels=y_true))
gd_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
accuracy=tf.reduce_mean(tf.cast(correct_mask,tf.float32))

with tf.Session() as sess:
    #train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs,batch_ys=data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step,feed_dict={x:batch_xs,y_true:batch_ys})
    ans=sess.run(accuracy,feed_dict={x:data.test.images,
                                     y_true:data.test.labels})


print (ans)
