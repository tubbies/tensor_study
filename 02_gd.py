#!/usr/bin/python
# vim:ts=4:sw=4:et:cms=#%s:fdm=marker:fdl=0
# from sys import *;
# import os;
# https://www.tensorflow.org/api_guides/python/train
import tensorflow as tf;
import numpy      as np;

# Main Function
if __name__ == "__main__":
    x_data = np.float32(np.random.rand(2,100));
    y_data = np.dot([0.100,0.200],x_data) + 0.300;
    b = tf.Variable(tf.zeros([1]));
    W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0)); 
    # 1x2 randomly uniform dist. 
    y = tf.matmul(W,x_data) + b;

    loss = tf.reduce_mean(tf.square(y - y_data));
    opti = tf.train.GradientDescentOptimizer(0.5);
    train = opti.minimize(loss);

    init = tf.initialize_all_variables();
    sess = tf.Session();
    sess.run(init);

    for step in range(0,201):
        sess.run(train);
        if step % 20 == 0:
            print(step,sess.run(W),sess.run(b));

