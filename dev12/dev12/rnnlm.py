import tensorflow as tf
from tensorflow.contrib.slim.python.slim.queues import QueueRunners
import numpy as np
import scipy as sp
import sys
import os
import time

from matplotlib import pyplot as plt
from matplotlib import cm
from math import ceil

from base_model import BaseModel

class RNNLM(BaseModel):
  '''
        RECURRENT NEURAL NETWORK LANGUAGE MODEL Constructor
        Init:
            self:
            network_architecture = None
            name = None
            dir = None
            load_path = None
            debug_mode = 0
            seed = 100
  '''
  def __init__(self, network_architecture=None, name=None, dir=None, load_path=None, debug_mode=0, seed=100):

    BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, dir=dir, load_path=load_path, debug_mode=debug_mode)

    with self._graph.as_default():

      with tf.variable_scope('input') as scope:
        self._input_scope = scope

        # x shoul be the padded sentences, the dimension should actually be
        # sth like TOTLEN * 20
        # y should be the targets.. what targets?
        # intuitively it should be the NEXT WORD...
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])

        # seqlens is kept to mantain information about the length of each sentence
        self.seqlens = tf.placeholder(tf.int32, [None])

        # dropout and batch are for the network architecture
        self.dropout =  tf.Variable(tf.ones(dtype=tf.float32, shape=[]), trainable=False, name='dropout_rate')
        self.batch_size = tf.placeholder(tf.int32, [])

        # ? Few doubts: why use dropout and batch_size as input? they are not
        # an input.....

      with tf.variable_scope('model') as scope:
        self._model_scope = scope

        # Here _construct_network does the trick
        # It receives as input:
        # the set of sentences x, their length, the batch size, WD and the dropout probability
        # ? What is WD? Should be weight decay
        self.predictions, self.logits, reshaped_activation, inl2, softmaxl2 = self._construct_network(input=self.x,
                                                                seqlens=self.seqlens,
                                                                batch_size=self.batch_size,
                                                                WD=self.network_architecture['L2'],
                                                                keep_prob=self.dropout)


      init = tf.global_variables_initializer()
      self.sess.run(init)

      self._saver = tf.train.Saver(tf.global_variables())

      # Restores saved model parameters
      if load_path != None:
        arch_path = os.path.join(load_path, 'weights.ckpt')
        with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
          f.write('Restoring Model parameters from: '+arch_path+'\n')
        self._saver.restore(self.sess, arch_path)

  def _construct_network(self, input, seqlens, batch_size, WD=1e-6, keep_prob=1.0, cell_type='lstm'):
    '''
        CONSTRUCT NETWORK: Builds the Computational Graph

        Args:
          self
          input : x
          seqlens: sentences lengths
          batch_size
          WD = WEIGHT DECAY
          keep_prob = dropout probability

        Returns:
          predictions, logits
    '''
    #keep_prob=0.5
    initializer = self.network_architecture['initializer']

    with tf.variable_scope('Embeddings') as scope:
      # inputs contains the embeddings of the words in the sentences
      # Basically here your converting from words to one-hot
      # Note: n_in is in the config file: size of the vocabulary.
      embedding, inl2 = self._variable_with_weight_decay("word_embedding", [self.network_architecture['n_in'], self.network_architecture['n_hid']], self._seed, WD)
      self.word_embeddings = embedding
      inputs = tf.nn.embedding_lookup(embedding, input, name='embedded_data')

    with tf.variable_scope('RNN', initializer=initializer(self._seed), regularizer=tf.contrib.layers.l2_regularizer(WD)) as scope:
      # cell contains the LSTM cell.
      # the number of units is specified in the config file: n_hid (e.g. 180)
      # The ACTIVATION function is a hiperbolic tangent tanh.
      # note: also reuse has been added to access again to the BasicLSTMCell.
      if cell_type=='lstm':
          print '\nBuilding RNNLM with LSTM cells\n'
          cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_hid'],
                                            activation=tf.nn.tanh,
                                            forget_bias=1.0,
                                            state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
      elif cell_type=='gru':
          print '\nBuilding RNNLM with GRU cells\n'
          # dev12: switch to GRU cells
          cell = tf.contrib.rnn.GRUCell(num_units=self.network_architecture['n_hid'],
                                    activation=tf.nn.tanh,
                                    reuse=tf.get_variable_scope().reuse,
                                    )
      else:
          print '\nPlease, specify correct cell type\n'
          exit()
      # Adding dropout to prevent overfitting
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

      initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      # Building the RNN around the LSTM cell
      outputs, state = tf.nn.dynamic_rnn(cell,
                                         inputs,
                                         sequence_length=seqlens,
                                         dtype=tf.float32,
                                         initial_state=initial_state,
                                         parallel_iterations=32)

    with tf.variable_scope('softmax_output') as scope:
      # Adds L2 regularisation to the softmax output
      weights, softmaxl2 = self._variable_with_weight_decay(name = "weights",
                                                 shape = [self.network_architecture['n_hid'],
                                                        self.network_architecture['n_out']],
                                                 seed = self._seed,
                                                 wd = WD)

      biases  = self._variable_on_gpu('biases', [self.network_architecture['n_out']], tf.constant_initializer(0.1))

      activations = outputs

      # sigma = Wx + b
      logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.network_architecture['n_hid']]), weights), [batch_size, -1, self.network_architecture['n_out']]) + biases

      print '[_construct_network] logits construction: ', logits


      # softmax (Wx+b)
      predictions =tf.nn.softmax(logits, dim=-1, name='predictions')

    return predictions, logits, activations, inl2, softmaxl2

  def _construct_cost(self, targets, logits, seqlens, maxlen, is_training=False):
    ''' CONSTRUCT COST: Computes the loss

        Returns the softmax cross entropy
        which is defined as:
         - sum sum t_ic log P(y_ic)
        this has then to be normalised over the total number of words -> Normalised Cross Entropy

        Note: you have to mask the padding that has been added to the input sentences
        Essentially the mask is a binary mask:
            input : '33 44 55 0 0 0 0 '
            mask : '1 1 1 0 0 0 0'
    '''
    # Create a mast to zero out xent for inputs beyond sequence length
    mask = tf.sequence_mask(seqlens, maxlen, dtype=tf.float32)
    # Cost computation:fit
    # 1. The function sparse_softmax_cross_entropy_with_logits computes the softmax CE.
    # 2. You apply the mask by doing simple pairwise multiplication
    # 3. You sum over the no. of samples
    # Note: here normalisation is missing......
    #       It is probably implemented somewhere else
    '''Note: This can be extended to a range of different costs. Trying most of the already available ones may be good'''

    unmasked_loss= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='xentropy_per_example')
    #unmasked_loss= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='centropy_per_example')
    #import pdb; pdb.set_trace()


    #logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.network_architecture['n_hid']]), weights), [batch_size, -1, self.network_architecture['n_out']]) + biases
    #import pdb; pdb.set_trace()
    print '[_construct_cost] logits computation: ', logits

    cost = tf.reduce_sum(mask * unmasked_loss)

    if self._debug_mode > 1:
      tf.scalar_summary('XENT', cost)

    # During TRAINING you add the L2 loss as regularisation
    # Check with Andrey but it seems so...

    if is_training:

      # Normalised CE : NCE
      # You divide the CE by the total no. of words.
      # Note: you add the loss to the collection because
      #       you will need all the losses to compute the total
      #       loss at the end of the minibatches.
      # But keep in mind that this is only orientative. During training you
      # change the paramenters.
      cost = cost/tf.reduce_sum(tf.cast(seqlens, dtype=tf.float32))
      tf.add_to_collection('losses', cost)

      # The total loss is defined as the target loss plus all of the weight
      # decay terms (L2 loss).
      total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')

      #import pdb; pdb.set_trace()
      print '[construct loss] losses collection: ', tf.get_collection('losses')
      return cost, total_cost

    else:
      return cost


  def fit(self, valid_data_list, trn_data_list, learning_rate=1e-2, lr_decay=0.8, batch_size=50, dropout=1.0, optimizer=tf.train.AdamOptimizer,
   n_epochs=30, stimulated=0, regtype='mse', regweight=1e-4, imshape=14, hypersphere=0, transformation='high_pass', cell_type='lstm'):
    ''''
        FIT: Performs the NN training

    '''

    print 'Start RNNLM Training...Intial learning rate per sample: {0}'.format(learning_rate)

    print 'Training information: \n Learning rate: {0} \n Learning rate decay: {1} \n '.format(learning_rate, lr_decay)
    print 'Batch size: {0} \n Dropout: {1}\n Optimizer: {2}\n No. Max Epochs: {3}\n'.format(batch_size, dropout, optimizer, n_epochs)
    print 'Stimulated: {0}\n Regularisation Type: {1}\n Regularisation weight: {2}\n Imshape: {3}'.format(stimulated, regtype, regweight, imshape)
    print 'Hypersphere: {0}\n Trainsformation type: {1}\n'.format(hypersphere, transformation)


    # Number of training examples and batch size
    n_examples = trn_data_list[1].shape[0]
    n_batches = n_examples/batch_size

    print 'Number of training examples: {0}\n'.format(n_examples)

    with self._graph.as_default():
      temp = set(tf.all_variables())
      # global_step: keeps track of the number of batches seen so far
      global_step = tf.Variable(0, trainable=False, name='global_step')

      lr = learning_rate
      self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)

      # Getting the inputs and splitting them into batches
      with tf.variable_scope(self._input_scope, reuse=True) as scope:
        batch_size_tensor = tf.constant(value=batch_size, dtype=tf.int32, shape=[])
        self._construct_queue(trn_data_list, batch_size=batch_size, capacity=n_examples)
        self._train_queue_init(trn_data_list)
      # Construct Training model
      with tf.variable_scope(self._model_scope, reuse=True) as scope:
        self.trn_predictions, self.trn_logits, activations, inl2, softmaxl2 = self._construct_network(input=self.data_queue_list[1],
                                                                        batch_size=batch_size_tensor,
                                                                        seqlens=self.data_queue_list[2],
                                                                        WD=self.network_architecture['L2'],
                                                                        keep_prob=1.0,
                                                                        cell_type=cell_type)

      # Constructs the loss function:
      # Here we should instroduce the stimulation patterns terms
      ''' BUIILDING THE LOSS FUNCTION: NCE + L2 + Regularisation '''

      #reshaped_activations = _node_organisation(activations)
      if stimulated:
          reshaped_activations = _node_organisation(activations, batch_size=batch_size, imshape=imshape, hypersphere=0)
          #print '[A. Node Organisation] Activations reorganised in a {0} x {1} grid'.format(imshape,imshape)

          transformed_activations = _activation_transformation(reshaped_activations, transformation = transformation, batch_size=batch_size, imshape=imshape)

          #print '[B. Activation Transformation] Activations transformed with a high pass filter.'

          target = _activation_target(self.data_queue_list[1], regtype, imshape=imshape)
          #print '[C. Activation Target] We set G=0. The regularisation function becomes the Frobenius norm of the high pass filtered activations.'

          regularizer = _regularisation(transformed_activations, target, seqlens=self.data_queue_list[2], regtype=regtype, regweight=regweight, batch_size=batch_size)
          #print '[D. Regularisation] Adding regularisation on the activations to the loss function'


      # NCE + L2 regularisation
      # (is_training=True)
      #print '[building up] [_construct_cost] self.data_queue_list: ', self.data_queue_list
      trn_cost, total_cost = self._construct_cost(targets=self.data_queue_list[0],
                                                  logits=self.trn_logits,
                                                  maxlen=20,
                                                  seqlens=self.data_queue_list[2],
                                                  is_training=True)


      # Constructs the evaluation cost:  the function is always the same, what changes is that now it uses
      # valid_data_list instead of data_queue_list
      # and is_training is now set to False (default)
      evl_cost = self._construct_cost(targets=self.y, logits=self.logits, maxlen=np.max(valid_data_list[2]), seqlens=self.seqlens)

      # Builds the gradient computation
      train_op = self._construct_train_op(total_cost, optimizer, None, None, batch_size, global_step, n_examples)

      #Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
      init=tf.variables_initializer(set(tf.global_variables()) - temp)

      '''STARTING THE SESSION HERE'''

      self.sess.run(init)

      #Create Summary ops and summary writer ### UNUSED!!! ###
      if self._debug_mode > 1:
        summary_op = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(self._dir, self._graph)

      # Start the Queue Runner
      tf.train.start_queue_runners(sess=self.sess)

      # Update Log with training details
      #import pdb; pdb.set_trace()
      with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
        if stimulated:
            format_str = ('Stimulated Framework: True\n Regularisation weight: '+str(regweight))
            format_str1 = ('\n regweight= ' +str(regweight)+ '\n imshape= ' + str(imshape)+'\n')
            f.write(format_str)
            f.write(format_str1)

	    print(format_str)
        else:
            format_str = ('Stimulated Framework: False\n')
            f.write(format_str)
        format_str = ('Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nOptimizer: %s')
        f.write(format_str % (lr, lr_decay, batch_size,  str(optimizer))+'\n\n')

      #format_str = ('Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, (%.1f examples/sec; %.3f ' 'sec/epoch')
      format_str = ('Epoch \t%d Alpha: %f Train Entropy: \t%.2f  Valid Entropy: %.2f  PPL: \t %.2f \t (%.1f examples/sec; %.3f ' 'sec/epoch')
      #format_str = ('Epoch \t %d \t Train Entropy: \t %.2f  \t Valid Entropy: \t %.2f  \t (%.1f examples/sec; %.3f ' 'sec/epoch')

      ce=[]
      wdin=[]
      wdsoftmax=[]
      R= []
      batch_ce=0
      batch_wdin=0
      batch_wdsoftmax=0
      batch_R=0

      start_time = time.time()
      old_eval_loss = 1000000.0
      decay = False
      for epoch in xrange(1, n_epochs+1):
        loss = 0.0
        batch_time = time.time()

        tot_batch_eval_loss=0

        for batch in xrange(n_batches):
          # This computes the gradients and the loss value for every batch
          # ? Where do we pass the actual data though? We only feed the dropout
          _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout : dropout})


          # Storing CE, L2 and R curves
          batch_ce+=loss_value
          batch_wdin+=self.sess.run(inl2)*1e6
          batch_wdsoftmax+=self.sess.run(softmaxl2)*1e6

          if stimulated:
              batch_R+=self.sess.run(regularizer)

          #if batch %100 == 0:
          #  print '[rnnlm] {0} batch of {1}'.format(batch, n_batches)

          #print time.time() -t
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          loss+=loss_value


          # Evaluating loss on the validation data for the current batch.
          # Just for training purposes
        #   i=batch
        #   if i<n_batches-1:
        #     batch_eval_loss = self.sess.run(evl_cost, feed_dict={self.y : valid_data_list[0][i*batch_size:(i+1)*batch_size], self.x : valid_data_list[1][i*batch_size:(i+1)*batch_size], self.seqlens : valid_data_list[2][i*batch_size:(i+1)*batch_size], self.batch_size : batch_size})
        #     #print '[rnnlm] [NORMALISED BATCH EVAL LOSS]: ', batch_eval_loss/valid_data_list[2][i*batch_size:(i+1)*batch_size][1]
        #     i+=1
        duration = time.time() - batch_time

        # Training per batch normalisation
        loss/=n_batches

        ce.append(batch_ce/batch_size) # this should add the value of the cross entropy to the list
        wdin.append(batch_wdin/batch_size)
        wdsoftmax.append(batch_wdsoftmax/batch_size)
        if stimulated:
            R.append(batch_R/batch_size)

        print  '[rnnlm] Training completed in {0} secs.'.format(duration)
        #print  '[rnnlm] Loss per batch: {0}'.format(loss)

        examples_per_sec = batch_size / duration
        sec_per_epoch = float(duration)

        #print  '[rnnlm] Evaluating perplexity...'
        #print '[rnnlm] Validation data list: ', sum(valid_data_list[2])

        eval_loss=0.0

        for i in xrange(len(valid_data_list[0])/50):
        #  print '[rnnlm] Currently evaluating..{0} of {1}'.format(i, len(valid_data_list))
        #  #orig: eval_loss += self.sess.run(evl_cost, feed_dict={self.y : valid_data_list[0][i*batch_size:(i+1)*batch_size], self.x : valid_data_list[1][i*batch_size:(i+1)*batch_size], self.seqlens : valid_data_list[2][i*batch_size:(i+1)*batch_size], self.batch_size : batch_size})
          eval_loss += self.sess.run(evl_cost, feed_dict={self.y : valid_data_list[0][i*50:(i+1)*50], self.x : valid_data_list[1][i*50:(i+1)*50], self.seqlens : valid_data_list[2][i*50:(i+1)*50], self.batch_size : 50})

        eval_loss /= np.sum(valid_data_list[2])

        if (eval_loss >= old_eval_loss) or decay == True:
          lr /=2.0
          assign_op = self.lr.assign(lr)
          self.sess.run(assign_op)
          decay = True
        old_eval_loss = eval_loss

        ''' SAVING '''

        # General logs
        with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
          f.write(format_str % (epoch, lr, loss, eval_loss, np.exp(eval_loss), examples_per_sec, sec_per_epoch)+'\n')
          save_curve('inputL2', wdin,self._dir)
          save_curve('softmaxL2', wdsoftmax,self._dir )
          save_curve('ce', ce, self._dir)

        # Stimulated Learning specific logs
        if stimulated:
            save_evolution(self.sess, epoch,reshaped_activations, os.path.join(self._dir, 'raw_activations'))
            save_evolution(self.sess, epoch,transformed_activations, os.path.join(self._dir, 'filtered_activations'))

        print (format_str % (epoch, lr, loss, eval_loss, np.exp(eval_loss),  examples_per_sec, sec_per_epoch))
        self.save()

      duration = time.time() - start_time
      with open(os.path.join(self._dir, 'LOG.txt'), 'a') as f:
          format_str = ('Training took %.3f sec')
          f.write('\n'+format_str % (duration)+'\n')
          f.write('----------------------------------------------------------\n')
      print (format_str % (duration))

  def predict(self, X):
    batch_size=5
    with self._graph.as_default():
      for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print var.op.name
      test_loss=0.0
      evl_cost = self._construct_cost(targets=self.y, logits=self.logits, maxlen=np.max(X[2]), seqlens=self.seqlens)

      predictions=None

      for i in xrange(len(X[0])/batch_size):
        test_loss_batch, batch_logits, batch_predictions = self.sess.run([evl_cost, self.logits, self.predictions], feed_dict={self.y : X[0][i*batch_size:(i+1)*batch_size], self.x : X[1][i*batch_size:(i+1)*batch_size], self.seqlens : X[2][i*batch_size:(i+1)*batch_size], self.batch_size : batch_size})
        test_loss+=test_loss_batch

        if predictions is None:
            predictions=batch_predictions
        else:
            predictions= np.concatenate([predictions, batch_predictions], axis=0)

        # If you want to save also the logits
        # if logits is None:
        #     logits=batch_logits
        # else:
        #     logits= np.concatenate([logits, batch_logits], axis=0)

      test_loss /= np.sum(X[2])

      ## Saving predictions and probabilities for N best list rescoring
      np.save(self._dir+'/predictions', predictions)
      print 'Test PPL', np.exp(test_loss)

'''############################################STIMULATED LEARNING#############################################################'''
def _node_organisation(activations, batch_size=64, bptt=20, imshape=14, hypersphere=0,save_folder='raw_activations'):
  '''
  A) NODE ORGANISATION: Reshapes the network activations into a 2D grid

      input:
        activations,
        batch_size=64,
        bptt=20,
        imshape=14,
        hypersphere=0,
        save_folder='raw_activations'

      output: reshaped_activations
              (note: sizes should be (batch_size * bptt * [shape, shape]) )
  '''
  reshaped_activations =[]
  reshaped_activations_in_sentence=[]
  i=0
  # we iterate the sentences in tha batch
  reshaped_activations=tf.reshape(activations, shape=[-1,imshape, imshape, 1])

  if hypersphere:
      print 'ORGANISATION: Hypersphere'
      reshaped_activations=hypersphere(reshaped_activations)
  else:
      print 'ORGANISATION: Flat Grid'

  return reshaped_activations

def hypersphere(reshaped_activations, imshape=14):
    # Rearranges the activation grids into donuts
    # using the Karnaugh Mapping
    trial=tf.tile(reshaped_activations, [1,3,3,1])

    return trial[:,imshape-1:2*imshape+1,imshape-1:2*imshape+1,:]

def _activation_transformation(reshaped_activations, transformation = 'high_pass', batch_size=64, bptt=20, imshape=14, save_folder='hp_activations'):
    '''
    B) ACTIVATION TRANSFORMATION: Applies a transformation to the activation patterns
        E.g.
            1. High pass filter
            2. Spheric High Pass Filter
            3. Probability Mass Function

        input: sess, reshaped_activations, transformation, batch_size, bptt, imshape
        output: transformed_activations
                (note: sizes should be (batch_size * bptt * [shape, shape]) )
    '''
    transformed_activations=[]
    if transformation=='high_pass':
        print 'Transformation: HIGH PASS'
        K = np.array([[-1,-1,-1],
                     [-1,  8, -1],
                     [-1, -1, -1]],  dtype=np.float32)
        K /= 8
        w = tf.constant(K)
        w = tf.constant(K)
        w = tf.reshape(w, (3,3,1,1))
        transformed_activations = tf.nn.conv2d(reshaped_activations, w, strides=[1, 1, 1, 1], padding='VALID')
    if transformation == 'pmf':
        print 'Transformation: PROBABILTY MASS FUNCTION'
        transformed_activations = reshaped_activations / tf.reduce_sum(reshaped_activations)
        #import pdb; pdb.set_trace()
    return transformed_activations

def _activation_target(words,target_type, imshape=14, batch_size=64, bptt=20):
    '''
        C) ACTIVATION TARGET: specify what you would the activation to be like
            input:
            output: target

            Notes: currently we just return G=0
    '''

    if target_type=='mse':
        print 'Target: ZEROS'
        target = tf.zeros([batch_size*bptt, imshape,imshape, 1])

    if target_type=='cos':
        print 'Target: POS maps'
        targets=np.load('posTargets.npy')
        tftargets=tf.Variable(targets)
        input=words
        targs = tf.nn.embedding_lookup(tftargets, input, name='targets')
        target = targs

    return target

def _regularisation(transformed_activations, target, seqlens, regtype='mse' ,regweight=1e-4,  type=1, batch_size=64, bppt=20):
    '''
        D) REGULARISATION: we compute the distance between the reshaped_activations
            and the targets with the Frobenius norm and add it to the collection of
            losses.

            Types:
                1. Mean Squared Error
                2. Cosine Distances
            input: reshaped_activations, target
            output: ?
    '''
    # 1. Mean Squared Error
    if regtype=='mse':
        print 'Regularisation: MEAN SQUARED ERROR'
        weight=regweight
        regularizer=tf.constant(0.0, dtype=tf.float32)
        loss = tf.nn.l2_loss(transformed_activations)*2/tf.reduce_sum(tf.cast(seqlens, dtype=tf.float32))
        tf.add_to_collection('losses', tf.multiply(weight,loss))
        regularizer = loss

    #2. Cosine Distances
    if regtype=='cos':
        print 'Regularisation: COSINE DISTANCES'
        normed_activation = tf.nn.l2_normalize(tf.reshape(transformed_activations, shape=[-1, 196]), dim=1)
        normed_target =  tf.nn.l2_normalize(tf.reshape(target, shape=[-1, 196]), dim=1)
        regularizer = tf.losses.cosine_distance(normed_activation, normed_target,dim=1)
        tf.add_to_collection('losses', tf.multiply(regweight,regularizer))
    return regularizer
'''#####################################################END######################################################################'''

'''##### ##### ###  VISUALISATION  ### ##### #####'''
def save_evolution(sess, epoch, word_activation, folder):
    # Saves the activations for each epoch in a numpy file

    if not os.path.exists(os.path.join(os.getcwd(), folder)):
        os.mkdir(folder)
    activations=sess.run(word_activation)

    np.save(folder+'/'+str(epoch), activations)
    return

def save_curve(filename, curve, folder):
    fp = open(folder+'/'+filename+'.txt', 'w')
    for value in curve:
        fp.write(str(value)+'\n')
    return
