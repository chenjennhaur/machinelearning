# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import math
  
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
  
batch_size = 128
num_steps = 3001

image_size = 28
num_labels = 10
# too small causes overfitting, too big causes underfitting
beta = 1e-5 
dropout = 0.75
#lower the learn rate, the slower it learns, but also the more accurate
start_learn_rate = 0.3
# if staircase_learn = num_steps, no effect. 
staircase_learn = 100

graph = tf.Graph()
with graph.as_default():

    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.

    #tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    #tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    x = tf.placeholder(tf.float32,shape=(None,image_size * image_size),name='inputs')
    y_ = tf.placeholder(tf.float32, shape=(None,num_labels),name='y-labels')
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    l_step = tf.Variable(0)  # count the number of steps taken.
    #learning_rate = tf.placeholder(tf.float32, shape=[])

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random valued following a (truncated)
    # normal distribution. The biases get initialized to zero.
    hidden1 = 1024
    hidden2 = 1024
    weights_h1 = tf.Variable(tf.truncated_normal([image_size*image_size,hidden1],stddev=1.0/math.sqrt(float(784))),name='hidden_w')
    biases_h1 = tf.Variable(tf.zeros([hidden1]),name='hidden_b')
    with tf.name_scope('hidden_layer_1'):
      hidden_l1 = tf.nn.relu(tf.matmul(x,weights_h1)+biases_h1)
      hidden_l1_d = tf.nn.dropout(hidden_l1,keep_prob) 
          
    weights_h2 = tf.Variable(tf.truncated_normal([hidden1,hidden2],stddev=1.0/math.sqrt(float(784))),name='hidden2_w')
    biases_h2 = tf.Variable(tf.zeros([hidden2]),name='hidden2_b')
    
    with tf.name_scope('hidden_layer_2'):
      hidden_l2 = tf.nn.relu(tf.matmul(hidden_l1_d,weights_h2)+biases_h2)
      hidden_l2_d = tf.nn.dropout(hidden_l2,keep_prob)     
    
    weights_ol = tf.Variable(tf.truncated_normal([hidden2, num_labels],stddev=1.0/math.sqrt(float(784))),name='output_w')
    biases_ol = tf.Variable(tf.zeros([num_labels]),name='output_b')
    with tf.name_scope('output_layer'):
      logits = tf.matmul(hidden_l2_d,weights_ol) + biases_ol
      y = tf.nn.softmax(logits)     
  
    #weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]),name='weight')
    #biases = tf.Variable(tf.zeros([num_labels]),name='bias')
    #y = tf.matmul(x, weights) + biases
        
    h_weight = tf.histogram_summary("weights_hidden", weights_h1)
    h_bias = tf.histogram_summary("bias_hidden", biases_h1)
    o_weight = tf.histogram_summary("weights_output", weights_ol)
    o_bias = tf.histogram_summary("bias_output", biases_ol)
    #_ = tf.histogram_summary("tr_pred",train_prediction)
    #_ = tf.histogram_summary("v_pred",valid_prediction)
    
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.

    with tf.name_scope("cost_function") as scope:
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_))
      loss_train = tf.scalar_summary("loss",loss)
      #Regularization 
      loss += beta * (tf.nn.l2_loss(weights_ol)+tf.nn.l2_loss(weights_h1)+tf.nn.l2_loss(weights_h2))
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    
    
    with tf.name_scope("train") as scope:
      #Constantly reduce learning rate
      #learning_rate = tf.train.exponential_decay(start_learn_rate,l_step,num_steps,0.95)
      #Reduce Learning Rate every 100 steps
      learning_rate = tf.train.exponential_decay(start_learn_rate,l_step,staircase_learn,0.95,staircase=True)
      l_r = tf.scalar_summary('lrn_rate',learning_rate)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=l_step)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    with tf.name_scope('test'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy = tf.scalar_summary('train_accuracy', accuracy)
    valid_accuracy = tf.scalar_summary('valid_accuracy', accuracy)
      
    
    #train_prediction = tf.nn.softmax(y)
    #valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
    #For 1 layer only
    #def u_accuracy(predictions, labels):
    #  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])
    #valid_pred = tf.matmul(tf.nn.relu(tf.matmul(valid_dataset,weights_h1)+biases_h1),weights_ol)+biases_ol
    #test_pred = tf.matmul(tf.nn.relu(tf.matmul(test_dataset,weights_h1)+biases_h1),weights_ol)+biases_ol


with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  #merged = tf.merge_all_summaries()
  merged = tf.merge_summary([train_accuracy,loss_train,l_r,h_weight,o_weight,h_bias,o_bias])
  writer = tf.train.SummaryWriter("/tmp/neural",session.graph_def)
  
  tf.initialize_all_variables().run()
  saver = tf.train.Saver(var_list={"w_h1": weights_h1,"b_h1":biases_h1,"w_ol":weights_ol,"b_ol":biases_ol})
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
    
    
    _, l, summary_op= session.run([optimizer, loss, merged],feed_dict=feed_dict)
    writer.add_summary(summary_op,step)
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      #tr_dict = {x : train_dataset, y_ : train_labels}
      tr_accy, t_summ = session.run([accuracy,train_accuracy],feed_dict=feed_dict)
      writer.add_summary(t_summ,step)
      print('Training accuracy: %.3f' % tr_accy)
      
      v_dict = {x : valid_dataset, y_ : valid_labels, keep_prob : 1}
      v_accy,v_summ = session.run([accuracy,valid_accuracy],feed_dict=v_dict)
      writer.add_summary(v_summ,step)
      print('Validation accuracy: %.3f' % v_accy)
      #print('Validation accuracy(udacity): %.3f' % u_accuracy(valid_pred.eval(),valid_labels))
      #print(v_accy)     
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      
  te_dict = {x : test_dataset, y_ : test_labels, keep_prob : 1}
  te_accy = session.run([accuracy,y],feed_dict=te_dict)
  print('Test accuracy: %.3f' % te_accy[0])
  save_path = saver.save(session,"/tmp/neural.ckpt")
  #print('Test accuracy(udacity): %.3f' % u_accuracy(test_pred.eval(),test_labels))
  