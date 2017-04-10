#!/usr/bin/python
# vim:ts=4:sw=4:et:cms=#%s:fdm=marker:fdl=0
# from sys import *;
# import os;
import tensorflow as tf;

# Main Function
if __name__ == "__main__":
    hello = tf.constant("Hello Tensorflow");
    a     = tf.constant(10,name="a");
    b     = tf.constant(20,name="b");
    sess  = tf.Session();
    print(sess.run(hello));
    print(sess.run(a+b));

