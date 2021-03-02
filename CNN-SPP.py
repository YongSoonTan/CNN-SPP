def CNNSPP(D,DT,F,model):
  import scipy.io as sio
  import time
  import os
  import math
  import numpy as np
  import matplotlib.pyplot as plt


  Dataset = D
  if DT == 'org':
    data_type = 'original'
  else:
    data_type = 'augmented'

  fs = model.fs
  batch_size = model.batch_size
  learn_rate = model.learn_rate
  out_pool_size1 = model.out_pool_size1
  out_pool_size = model.out_pool_size
  num_layers = model.num_layers
  fc_nodes = model.fc_nodes
  conv_block_1_fm = 64
  conv_block_2_fm = 128

  if F == 1:
    file_name = '1st_fold'
  elif F == 2:
    file_name = '2nd_fold'
  elif F == 3:
    file_name = '3rd_fold'
  elif F == 4:
    file_name = '4th_fold'
  elif F == 5:
    file_name = '5th_fold'
  path = os.path.join('CrossVal', 'D'+Dataset)
  print("path " ,path)
  if data_type == 'original':
    Train =sio.loadmat(os.path.join(path, 'D'+Dataset+'_'+file_name+'_train.mat'))
  else:
    Train =sio.loadmat(os.path.join(path, 'Augmented_D'+Dataset+'_'+file_name+'_train.mat'))
  Test = sio.loadmat(os.path.join(path, 'D'+Dataset+'_'+file_name+'_test.mat'))

  if Dataset == '1':
    number_of_classes = 24
    num_of_ep = 50
    if data_type == 'augmented':
      train_imgs = 526190
    else:
      train_imgs = 52619
    iteration = math.ceil((num_of_ep * train_imgs) / batch_size)
  elif Dataset == '2':
    number_of_classes = 36
    num_of_ep = 200
    if data_type == 'augmented':
      train_imgs = 20120
    else:
      train_imgs = 2012
    iteration = math.ceil((num_of_ep * train_imgs) / batch_size)
  else:
    number_of_classes = 10
    num_of_ep = 200
    if data_type == 'augmented':
      train_imgs = 16000
    else:
      train_imgs = 1600
    iteration = math.ceil((num_of_ep * train_imgs) / batch_size)

  iteration_to_display = int(iteration / 16)
  list_to_display = []
  for i in range(16):
      if i !=16:
          list_to_display.append(int(iteration_to_display*(i+1)))
  del i


  num = 0
  num1 = 0
  for i in range( len(out_pool_size)):
      num += out_pool_size[i]**2
  for i in range( len(out_pool_size1)):
      num1 += out_pool_size1[i]**2  

  SPP_C_nodes = ((num*conv_block_1_fm) + (num1*conv_block_2_fm))


  Train_Images = Train['trainImages']
  Train_Labels = Train['trainLabels2']
  total_trainImages = len(Train_Images[0,2])
  print(total_trainImages)
  Train_Images = Train_Images.reshape(784,total_trainImages).transpose().astype('float32')
  Train_Labels = Train_Labels.transpose().astype('float64')


  Test_Images = Test['testImages']
  Test_Labels = Test['testLabels2']
  total_testImages = len(Test_Images[0,2])
  Test_Images = Test_Images.reshape(784,total_testImages).transpose().astype('float32')
  Test_Labels = Test_Labels.transpose().astype('float64')
  Target_labels = np.argmax(Test_Labels,axis=1)

  del Test
  del Train

  import tensorflow as tf
  slim = tf.contrib.slim
  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(1)

    def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size, scope="SPP_1"):
      with tf.name_scope(scope):
        with tf.variable_scope(scope) :
          for i in range(len(out_pool_size)):
            h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
            w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
            pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
            pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
            new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
            max_pool = tf.nn.max_pool(new_previous_conv,
                           ksize=[1,h_size, h_size, 1],
                           strides=[1,h_strd, w_strd,1],
                           padding='SAME')
            if (i == 0):
              spp = tf.reshape(max_pool, [num_sample, -1])
            else:
              spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])

          return spp

    def weight_variable(shape,n):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial,name=n)

    def bias_variable(shape,n):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial,name=n)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def conv_layer(input, fsize, in_fm, out_fm, training, name="conv_lyr"):
      with tf.name_scope(name):
        with tf.variable_scope(name) :
          w = weight_variable([fsize, fsize, in_fm, out_fm],"w"+name)
          b = bias_variable([out_fm],"b"+name)
          conv = conv2d(input, w) + b
          BNconv = tf.layers.batch_normalization(conv, training=training)
          act = tf.nn.relu(BNconv)
          return act

    def conv_block(input, fsize, in_fm, out_fm, layers, training, name="conv_block"):
      with tf.name_scope(name):
        with tf.variable_scope(name) :
          current = input
          for idx in range(layers): 
            current = conv_layer(current, fsize, in_fm, out_fm, training, name="conv_lyr_"+str(idx+1))
            in_fm = out_fm
          return current

    def fully_connected_lyr(input, input_nodes, output_nodes, name="FC_lyr"):
      with tf.name_scope(name):
        W_fc = weight_variable([input_nodes, output_nodes],"w_fc")
        b_fc = bias_variable([output_nodes],"b_fc")
        out = tf.matmul(input, W_fc) + b_fc
        out = tf.layers.batch_normalization(out, training=training, axis=1)
        out = tf.nn.relu(out)
        out_drop = tf.nn.dropout(out, keep_prob, seed=1)
        return out_drop

    def FC_lyr(input, input_nodes, fc_nodes):
      for i in range(len(fc_nodes)):
        if i == 0:
          input = fully_connected_lyr(input, input_nodes, fc_nodes[i], name='FC_lyr'+str(i+1))
        else: 
          input = fully_connected_lyr(input, fc_nodes[i-1], fc_nodes[i], name='FC_lyr'+str(i+1))
      return input


    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, number_of_classes])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)


    out_cb1 = conv_block(x_image, fs, 1, conv_block_1_fm, num_layers, training, name="conv_block_64")
    out_cb1_spp = spatial_pyramid_pool(out_cb1, tf.shape(out_cb1)[0], [int(out_cb1.get_shape()[1]), int(out_cb1.get_shape()[2])], out_pool_size, scope="SPP_1")
    b1_conv_printop = tf.Print(out_cb1, [out_cb1])
    out_cb1 = max_pool_2x2(out_cb1)

    out_cb2 = conv_block(out_cb1, fs, conv_block_1_fm, conv_block_2_fm, num_layers, training, name="conv_block_128")
    out_cb2_spp = spatial_pyramid_pool(out_cb2, tf.shape(out_cb2)[0], [int(out_cb2.get_shape()[1]), int(out_cb2.get_shape()[2])], out_pool_size1, scope="SPP_2")
    b2_conv_printop = tf.Print(out_cb2, [out_cb2])

    with tf.name_scope("SPP_concatenation"):
      SPP_C = tf.concat(axis=1, values=[out_cb1_spp, out_cb2_spp] )
      SPP_C = tf.nn.dropout(SPP_C, keep_prob, seed=1)

    with tf.name_scope("fc_lyr"):
      out_fc = FC_lyr(SPP_C, SPP_C_nodes, fc_nodes)

    with tf.name_scope("last_fc_lyr"):
      W_fc3 = weight_variable([int(out_fc.get_shape()[1]), number_of_classes],"w_fc3")
      b_fc3 = bias_variable([number_of_classes],"b_fc3")
      y_conv = tf.matmul(out_fc, W_fc3) + b_fc3
      prediction_prob = tf.nn.softmax(y_conv)
      prediction_prob_printop = tf.Print(prediction_prob, [prediction_prob])

    with tf.name_scope("Xent"):
       cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    with tf.name_scope("train"):
      extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(extra_update_ops):
         train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      wrong_prediction = tf.not_equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      wrong_prediction_printop = tf.Print(wrong_prediction, [wrong_prediction])
      predicted_labels = tf.argmax(y_conv, 1)
      predicted_labels_printop = tf.Print(predicted_labels, [predicted_labels])


    index = 0
    index_end = index + batch_size
    remaining = 0
    start_time = time.time()
    costs = []
    accuracy_list = []
    list_of_predicted_list = []


    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer(),tf.set_random_seed(0))
      for i in range(iteration):
        if index_end > total_trainImages:
                  remaining = total_trainImages - (index_end-batch_size)  
                  images = Train_Images[(index_end-batch_size):total_trainImages, :]
                  labels = Train_Labels[(index_end-batch_size):total_trainImages, :]
                  index = 0
                  index_end = index + batch_size - remaining
                  images = np.vstack((images, Train_Images[index:index_end, :]))
                  labels = np.vstack((labels, Train_Labels[index:index_end, :]))
                  batch = (images, labels)
                  index = index_end
                  index_end = index + batch_size
        else:
                  batch = (Train_Images[index:index_end, :], Train_Labels[index:index_end, :])
                  index = index + batch_size 
                  index_end = index_end + batch_size

        if i in list_to_display:
          elapsed_time = time.time() - start_time
          print('Elapsed Time Before for loop: %f secs' % elapsed_time)
          Accuracy = 0
          itrt_index = i

          if Dataset == '1':
            if file_name == '5th_fold':
              num_test = 13154
            else:
              num_test = 13155
          elif Dataset == '2':
            num_test = 503
          elif Dataset == '3':
            num_test = 400
          print(num_test)

          for img_index in range(num_test):
            t_image = np.array(Test_Images[img_index,:]).reshape(1,784)
            t_label = np.array(Test_Labels[img_index,:]).reshape(1,number_of_classes)
            test_acc = accuracy.eval(feed_dict={
                x: t_image, y_: t_label,
                keep_prob: 1.0, training:False})
            Accuracy += test_acc
            wrong, predicted, prediction_prob = sess.run([wrong_prediction_printop, 
                                 predicted_labels_printop,prediction_prob_printop], 
                                feed_dict={
                x: t_image, y_: t_label, 
                keep_prob: 1.0, training:False})
            if img_index <= 3:
              b1, b2 = sess.run([b1_conv_printop, b2_conv_printop], 
                                  feed_dict={
                  x: t_image, y_: t_label, 
                  keep_prob: 1.0, training:False})
              if img_index == 0:
                b1_list = b1
                b2_list = b2
              else:
                b1_list = np.append(b1_list,b1,axis=0)
                b2_list = np.append(b2_list,b2,axis=0)
  
            if img_index == 0 :
              wrong_list_1 = wrong
              predicted_list_1 = predicted
              prediction_prob_1 = prediction_prob
            else:
              wrong_list_1 = np.append(wrong_list_1,wrong,axis=0)
              predicted_list_1 = np.append(predicted_list_1,predicted,axis=0)
              prediction_prob_1 = np.append(prediction_prob_1, prediction_prob)


          Accuracy = Accuracy/num_test
          accuracy_list.append(Accuracy)
          list_of_predicted_list.append(predicted_list_1)
          print('Average test accuracy: %g' % Accuracy)
          epoch_around = math.ceil((itrt_index * batch_size) / total_trainImages)
          sio.savemat('D'+Dataset+'_'+file_name+'_'+str(epoch_around)+'ep_'+data_type+'_predicted_labels_list.mat', {'wrong_list':wrong_list_1, 'predicted_list': predicted_list_1, 'Target_labels':Target_labels,  
                                                                                                       'prediction_prob':prediction_prob, 'b1_list':b1_list, 'b2_list':b2_list })

          elapsed_time = time.time() - start_time
          print('Elapsed Time: %f secs' % elapsed_time)
          print('Batch Size & Iteration & Total Train Imgs : %d & %d & %d' % (batch_size, itrt_index, total_trainImages))   
          print('learning_rate : %f ' % learn_rate)     

          if 'out_pool_size' in locals():
            print('SPP_1 :', end=" ")
            for i1 in range(len(out_pool_size)):
              if i1 == len(out_pool_size)-1 :
                print (out_pool_size[i1])
              else:
                print (out_pool_size[i1], end=" & ")

          if 'out_pool_size1' in locals():
            print('SPP_2 :', end=" ")
            for i1 in range(len(out_pool_size1)):
              if i1 == len(out_pool_size1)-1 :
                print (out_pool_size1[i1])
              else:
                print (out_pool_size1[i1], end=" & ")

          print('data_type :', data_type)
          print('file_name :', file_name)
          print('fc_nodes :', end=" ")
          for i1 in range(len(fc_nodes)):
            if i1 == len(fc_nodes)-1 :
              print (fc_nodes[i1])
            else:
              print (fc_nodes[i1], end=" & ")

          epoch_around = (itrt_index * batch_size) / total_trainImages
          print('Number of epochs : %f ' % epoch_around)

          # plot the cost
          plt.plot(np.squeeze(costs))
          plt.ylabel('cost')
          plt.xlabel('iterations (per tens)')
          plt.title("Learning rate =" + str(learn_rate))
          plt.show()

        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], 
              keep_prob: 1.0, training:False})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        _, loss  = sess.run([train_step, cross_entropy], 
                                                   feed_dict={x: batch[0], y_: batch[1], 
                                                              keep_prob: 0.5, training:True})

        iteration_cost = 0                       # Defines a cost related to an epoch
        num_minibatches = int(total_trainImages / batch_size) # number of minibatches of size minibatch_size in the train set        
        iteration_cost += loss / num_minibatches
        costs.append(iteration_cost)
        if i % 100 == 0:
          print ('Loss: ',loss)

      Accuracy = 0
      training_time = time.time() - start_time
      print('Training Time: %f secs' % training_time)


      if Dataset == '1':
        if file_name == '5th_fold':
          num_test = 13154
        else:
          num_test = 13155
      elif Dataset == '2':
        num_test = 503
      elif Dataset == '3':
        num_test = 400
      print(num_test)

      for img_index in range(num_test):
        t_image = np.array(Test_Images[img_index,:]).reshape(1,784)
        t_label = np.array(Test_Labels[img_index,:]).reshape(1,number_of_classes)
        test_acc = accuracy.eval(feed_dict={
            x: t_image, y_: t_label,
            keep_prob: 1.0, training:False})
        Accuracy += test_acc
        wrong, predicted = sess.run([wrong_prediction_printop, predicted_labels_printop],  feed_dict={
            x: t_image, y_: t_label, 
            keep_prob: 1.0, training:False})
        if img_index == 0 :
          wrong_list = wrong
          predicted_list = predicted
        else:
          wrong_list = np.append(wrong_list,wrong,axis=0)
          predicted_list = np.append(predicted_list,predicted,axis=0)


      Accuracy = Accuracy/num_test
      print('Average test accuracy: %g' % Accuracy)
      accuracy_list.append(Accuracy)
      list_of_predicted_list.append(predicted_list)

      elapsed_time = time.time() - start_time
      print('Elapsed Time: %f secs' % elapsed_time)
      print('Batch Size & Iteration & Total Train Imgs : %d & %d & %d' % (batch_size, iteration, total_trainImages))   
      print('learning_rate : %f ' % learn_rate)

      if 'out_pool_size' in locals():
        print('SPP_1 :', end=" ")
        for i in range(len(out_pool_size)):
          if i == len(out_pool_size)-1 :
            print (out_pool_size[i])
          else:
            print (out_pool_size[i], end=" & ")

      if 'out_pool_size1' in locals():
        print('SPP_2 :', end=" ")
        for i in range(len(out_pool_size1)):
          if i == len(out_pool_size1)-1 :
            print (out_pool_size1[i])
          else:
            print (out_pool_size1[i], end=" & ")


      print('data_type :', data_type)
      print('file_name :', file_name)
      print('fc_nodes :', end=" ")
      for i1 in range(len(fc_nodes)):
        if i1 == len(fc_nodes)-1 :
          print (fc_nodes[i1])
        else:
          print (fc_nodes[i1], end=" & ")

      epoch_around = math.ceil((iteration * batch_size) / total_trainImages)
      if epoch_around == 51:
        epoch_around = 50
      print('Number of epochs : %f ' % epoch_around)

      # plot the cost
      plt.plot(np.squeeze(costs))
      plt.ylabel('cost')
      plt.xlabel('iterations (per tens)')
      plt.title("Learning rate =" + str(learn_rate))
      plt.show()


    sio.savemat('D'+Dataset+'_'+file_name+'_'+str(epoch_around)+'ep_'+data_type+'_predicted_labels_list.mat', {'wrong_list':wrong_list, 'predicted_list': predicted_list, 'Target_labels':Target_labels, 'accuracy_list':accuracy_list, 'list_of_predicted_list':list_of_predicted_list, 'costs':costs})
    
    
class MyModel:
  out_pool_size = [4,3,2] 
  out_pool_size1 = [4,3,2]
  num_layers = 4
  fs = 3
  batch_size = [16]
  fc_nodes = [1024]  #must be in bracket, add extra fc layer by adding number of nodes  [1024, 512], two fc layers, first is [SPP_C_nodes, 1024], second is [1024,512] and so on
  learn_rate = 0.001

model = MyModel()

CNNSPP('3','org',1,model)
CNNSPP('3','org',2,model)
CNNSPP('3','org',3,model)
CNNSPP('3','org',4,model)
CNNSPP('3','org',5,model)