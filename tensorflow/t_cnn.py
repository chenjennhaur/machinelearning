# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
  
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
  
#hyper-parameters
num_steps = 1001
beta = 5e-4 
dropout = 0.5
start_learn_rate = 0.5
num_hidden = 64

#CNN parameters
batch_size = 16
patch_size = 5
depth = 16

image_size = 28
num_labels = 10
num_channels = 1

graph = tf.Graph()
with graph.as_default():
  
    # Input data.
    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels),name='inputs')
    y_ = tf.placeholder(tf.float32, shape=(None, num_labels),name='y-labels')
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    global_step = tf.Variable(0)  # count the number of steps taken.
    
  
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([784, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        #padding=SAME, input = output
        with tf.name_scope('conv_layer_1'):
          conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
          relu = tf.nn.relu(conv + layer1_biases)
          pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        with tf.name_scope('conv_layer_2'):
          conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
          relu = tf.nn.relu(conv + layer2_biases)
          pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        
        reshape = tf.reshape(pool,[-1,784])
        with tf.name_scope('hidden'):
          hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
          hidden_d = tf.nn.dropout(hidden,keep_prob)
        return tf.matmul(hidden_d, layer4_weights) + layer4_biases
  
    # Training computation.
    with tf.name_scope('output_layer'):
      logits = model(x)
      y = tf.nn.softmax(logits)
    
    with tf.name_scope('cost_function'):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_))
      loss_summ = tf.scalar_summary("loss",loss)
      #Regularization 
      #loss += beta * (tf.nn.l2_loss(weights_ol)+tf.nn.l2_loss(weights_h1))
    
    with tf.name_scope("train") as scope:
      #learning_rate = tf.train.exponential_decay(start_learn_rate, global_step,num_steps,0.96,staircase=True)
      #lr = tf.scalar_summary('learning_rate',learning_rate)
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
      
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    with tf.name_scope('test'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = tf.scalar_summary('train_accuracy', accuracy)
    valid_accuracy = tf.scalar_summary('valid_accuracy', accuracy)
      
    

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 

  merged = tf.merge_summary([train_accuracy,loss_summ])
  writer = tf.train.SummaryWriter("/tmp/cnn",session.graph_def)
  
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    #Evaluate the tuples in the session.run([,,])

    # Generate a minibatch.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {x : batch_data, y_ : batch_labels, keep_prob : dropout}
    
    _, l, summary_op = session.run([optimizer, loss, merged],feed_dict=feed_dict)
    writer.add_summary(summary_op,step)
    if (step % 50 == 0):
      print('Loss at step %d: %f' % (step, l))
      #tr_dict = {x : train_dataset, y_ : train_labels}
      tr_accy, t_summ = session.run([accuracy,train_accuracy],feed_dict=feed_dict)
      writer.add_summary(t_summ,step)
      print('Training accuracy: %.3f' % tr_accy)
      
      v_dict = {x : valid_dataset, y_ : valid_labels, keep_prob : 1}
      v_accy,v_summ = session.run([accuracy,valid_accuracy],feed_dict=v_dict)
      writer.add_summary(v_summ,step)
      print('Validation accuracy: %.3f' % v_accy)
      
      #print(v_accy)     
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      
  te_dict = {x : test_dataset, y_ : test_labels, keep_prob : 1}
  te_accy = session.run([accuracy],feed_dict=te_dict)
  print('Test accuracy: %.3f' % te_accy[0])
  