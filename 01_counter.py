#!/usr/bin/python
# vim:ts=4:sw=4:et:cms=#%s:fdm=marker:fdl=0
# from sys import *;
# import os;
import tensorflow as tf;

# Main Function
if __name__ == "__main__":
    state = tf.Variable(0,name="count");
    one   = tf.constant(1);
    new_val = tf.add(state,one);
    update  = tf.assign(state,new_val);

    init_op = tf.initialize_all_variables(); # Initialize all variables in graph

    with tf.Session() as sess:
        sess.run(init_op);
        print(sess.run(state));
        for _ in range(3):
            sess.run(update);
            print(sess.run(state));


