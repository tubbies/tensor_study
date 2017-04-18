# TensorFlow

## Installation w/ virtualenv

- Google recommend to the virtualenv installation [1]

  1. Install pip and virtualenv 

     ```bash
     $ sudo apt-get install python-pip python-dev python-virtualenv 
     $ sudo apt-get install python3-pip python3-dev python3-virtualenv
     ```
  2. Create virtualenv environment
	```bash
     $ virtualenv --system-site-packages __target_directory__
     ```

  3. Activate virtualenv environment
	```bash
    # Assume that __target_directory__ is ~/tensorflow
    $ source ~/tensorflow/bin/activate       # For bash, ksh, zsh
    $ source ~/tensorflow/bin/activate.csh   # For csh, tcsh
    ```
    -- Your prompt will be changed when it is activated
    ```bash
    (tensorflow) $ 
    ```
    -- If you done using tensorflow, by typing **deactivate** to deactivate
    ```bash
    (tensorflow) $ deactivate
    $ 
    ```
    
  4. Install tensorflow using pip by typing one of followings
  	```bash
    (tensorflow) $ pip  install --upgrade tensorflow     # for python2.x w/o GPU(CUDA) support
    (tensorflow) $ pip3 install --upgrade tensorflow     # for python3.x w/o GPU(CUDA) support
    (tensorflow) $ pip  install --upgrade tensorflow-gpu # for python2.x w/  GPU(CUDA) support
    (tensorflow) $ pip3 install --upgrade tensorflow-gpu # for python3.x w/  GPU(CUDA) support
    ```
  5. Test
  ```bash
  (tensorflow) $ python
  >>> import tensorflow as tf
  ```

## Programming model and basic concepts[2]

- TensorFlow is an open source software library for numerical computation using data flow graphs.[3]
- Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.

#### Tensor 
- Centeral unit of data in tensorflow.
- Represented as multiple dimension array.
- A tensor's **rank** is the number of dimension. If tensor is composed of n-dimensional array, its rank is _n_. 
- In [caffe](https://github.com/BVLC/caffe) deep learning network, it is called as _blob_.

#### Operation
- Nodes in the graph.
- Each operation has zero or more inputs and outputs(Tensor).

#### Session
- A session encapsulates the control and state of the TensorFlow runtime.
- To actually evaluate the nodes, we must run the computational graph within a **session**.

#### Variables
- In tensorflow, the parameters of the model are typically stored in tensors held in variables.
- Variables usually represented as multi-dimensional array

#### The Computational Graph
- A Tensorflow computation is described by a directed _graph_, which is compoed of a set of _nodes_.


## Getting Started with Basic APIs

Tensorflow has APIs available in several languages such as [Python](https://www.python.org), [C++](https://en.wikipedia.org/wiki/The_C%2B%2B_Programming_Language), [Java](https://www.java.com), [Go](https://golang.org/). In those APIs, Python API is the most complete and the easist to use. 
Thus, In this guide, I will explain with Python APIs.

#### Hello World 

```python
import tensorflow as tf;
if __name__ == "__main__": # Hello World!
    hello = tf.constant("Hello Tensorflow");
    sess  = tf.Session();
    print(sess.run(hello));
```

- It is basic form of **Hello World** using tensorflow.

#### Basics

```python
>> node1 = tf.constant(3.0,tf.float32  ) 
>> node2 = tf.constant(4.0             ) # tf.float32 implicitly
>> node3 = tf.constant(5.0,name='node3') # can define name explicitly
>> print(node1,node2,node3)
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32) Tensor("node3:0", shape=(), dtype=float32)
```

_3.0, 4.0, 5.0_ is not shown because computational graph is not evaluated using session.

To evaluate Tensorflow runtime, We have to define and evaluation graph using **session**.

```python
>> sess = tf.Session()
>> print(sess.run([node1,node2,node3]))
[3.0, 4.0, 5.0]
```

We can build more compilcated computation using **operations**. 

```python
>> node3 = tf.add(node1, node2)        # Add two nodes
>> print("node3: ", node3)             # Print node3
node3 :  Tensor("Add:0", shape=(), dtype=float32)
>> print("sess.run(node3): ",sess.run(node3))
sess.run(node3): 7.0
```
Computational graph of this flow should be like this.
![tf.add](https://www.tensorflow.org/images/getting_started_add.png)

#### Running Graph

- **tf.Session** is class for running Tensorflow operations.

	###### Class Initialization member function. Usually leave it void.

	> __init__(target='', graph=None, config=None)

	###### **run**  member function : Runs operations and evaluate tensors in fetchs.
    
    > run(fetches, feed_dict=None, options=None, run_metadata=None)


#### Constant Values

- tf.constant : Create a constant in tensor.

	> tf.constant(value, dtype=None, shape=None, name='Const')

- tf.zeros : Create a tensor with zero vector.

	> tf.zeros(shape, dtype=tf.float32, name=None)

	```python
	>> v_zero = tf.zeros([2,3])
	>> sess = tf.Session()
	>> print(sess.run(v_zero))
	[[0, 0, 0]
 	 [0, 0, 0]]
	```

- tf.ones : Create a tensor with all elements set to '1'.

	> tf.ones(shape, dtype=tf.float32, name=None)

- tf.fill : Create a tensor with all elements set to given value __val__.

	> tf.fill(dims, value, name=None)

	```python
	>> v_zero = tf.fill([2,3],3)
	>> sess = tf.Session()
	>> print(sess.run(v_zero))
	[[3, 3, 3]
 	 [3, 3, 3]]
	```

#### Random Values

- tf.random_normal : Create Random normally distributed value with mean and standard deviation.

	> tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

- tf.random_uniform : Create uniformly distributed value with minimum and maximum values.

	> tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

- tf.random_gamma : Create gamma distribution

	> tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

- tf.set_random_seed : Sets a graph-level random seed.

    > tf.set_random_seed(seed)

#### Variables 

- **Variables** is class of tensorflow.
- Variables class maintain state(s) or parameter(s) in the graph.
- You have to _initialize_ your variable before the session execution.

- tf.Variables : Tensorflow graph that has multiple functionality

	it has initialization member function like this

	> tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None) {:#Variable.init}

	- Can assign variable with new value

	> tf.Variable.assign(value, use_locking=False)
		
    ```python
    >> W = tf.Variable(10)
    >> assign_op = W.assign(100)
    >> init = tf.global_variable_initializer()
    >> with tf.Session.as sess:
    >>     sess.run(init)
    >>     print(sess.run(W))
    10
    >> with tf.Session.as sess:
    >>     sess.run(init)
    >>     print(sess.run(assign_op))
    100
    ```

- tf.variables\_initializer : Initialize all variables in __var\_list__. 

	> variables_initializer(var_list, name='init')

	**tf.initialize\_variables** is equivalent but it is deprecated. Please use **tf.variables\_initializer**.
    
#### Placeholder 

- tf.placeholder : Inserts a placeholder for a tensor that will be always fed. It can be fed during session is running.

	> tf.placeholder(dtype, shape=None, name=None)

    ```python
    >> W = tf.Variable([ 0.3], tf.float32)
    >> b = tf.Variable([-0.3], tf.float32)
    >> x = tf.placeholder(tf.float32)
    >> linear_model = W * x + b
    >> init = tf.global_variables_initializer()
    >> sess = tf.Session()
    >> sess.run(init)
    >> print(sess.run(linear_model))    # Error : will fail because x was not fed.
    >> print(sess.run(linear_model, {x:[1,2,3,4]})) # Succeed
    [0, 0.300001, 0.600002, 0.9000004]
    ```
    
#### tf.train

- Tensorflow provides varios optimizers. The simplest optimizer is **gradient descent** optimizer that slowly change each variable in order to minimize loss function.

	```python
    >> import numpy as np
	>> import tensorflow as tf
	>> W = tf.Variable([.3], tf.float32)
	>> b = tf.Variable([-.3], tf.float32)
	>> x = tf.placeholder(tf.float32)
	>> y = tf.placeholder(tf.float32)
	>> linear_model = W * x + b
	>> loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
	>> optimizer = tf.train.GradientDescentOptimizer(0.01) # optimizer
	>> train = optimizer.minimize(loss) # training data
	>> x_train = [1,2,3,4]
	>> y_train = [0,-1,-2,-3]
	>> init = tf.global_variables_initializer()
	>> sess = tf.Session()
	>> sess.run(init) # reset values to wrong
	>> for i in range(1000):
	>>     sess.run(train, {x:x_train, y:y_train})
	>> curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
	>> print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
	W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
    ```
    
    ###### List of optimizer
	|Optimizer|
	|:-|
	|tf.train.GridiendDescentOptimizer|
	|tf.train.AdagradOptimizer|
	|tf.train.MomentumOptimizer|
	|tf.train.AdamOptimizer|
	|tf.train.ProximalGradientDescentOptimizer|
	|tf.train.ProximalAdagradOptimizer|
	|tf.train.RMSPropOptimizer|

#### Basic Operation
|Operation |Description||
|:---|:----|:--|
|**tf.add**| Add two or more tensors (matrix addition)| |
|**tf.mul**| Multiply two matrix in element wise||
|**tf.matmul**| Multiply two or more matrix (Real Matrix multiplication)| |
|**tf.cond**| Conditional statement | tf.cond(pred, fn1, fn2, name=None) |
|**tf.exp**| Exponential of _x_| |
|**tf.log**| Natural Logarithm of _x_ | | 
|**tf.pow**| Compute power of _x_ and _y_ ||
|**tf.sqrt**| Compute square root of _x_ ||


- Example
	
	```python 
	>> tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y)) # if x<y, return x+z if not, return y**2
	```



#### Simple Examples - Linear Regression

```python
>> import numpy as np
>> import tensorflow as tf
>> import matplotlib.pyplot as plt

>> W = tf.Variable([.3], tf.float32)
>> b = tf.Variable([-.3], tf.float32)
>> x = tf.placeholder(tf.float32)
>> linear_model = W * x + b
>> y = tf.placeholder(tf.float32)
>> loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
>> optimizer = tf.train.GradientDescentOptimizer(0.01)
>> train = optimizer.minimize(loss)
>> x_train = [1,2,3,4]
>> y_train = [0,-1,-2,-3]
>> init = tf.global_variables_initializer()
>> sess = tf.Session()
>> sess.run(init) # reset values to wrong
>> w_cont = [];
>> b_cont = [];

>> for i in range(1000):
>>    sess.run(train, {x:x_train, y:y_train})
>>    curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
>>    w_cont.append(curr_W);
>>    b_cont.append(curr_b);
>>    #print("[%04d] W: %s b: %s loss: %s"%(i,curr_W, curr_b, curr_loss))
>> plt.plot(range(1,1001,1),w_cont);
>> plt.plot(range(1,1001,1),b_cont);
>> plt.show();
```



## Tensorboard

- Can visualize your learning
- Generate report by summary operation 
- Can visualize **tensorboard** command in your command line (not python)


[1] https://www.tensorflow.org/install/install_linux
[2] https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf
[3] https://www.tensorflow.org/
[4] https://www.tensorflow.org/versions/master/api_docs/python/
[5] https://web.stanford.edu/class/cs20si

[TensorFlow 시작하기.md](https://gist.github.com/haje01/202ac276bace4b25dd3f)

[TensorFlow Documentation](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/tutorials/mnist/beginners/)
