import tensorflow as tf
from PIL import Image
from scipy.misc import imread, imshow,imresize
from glob import glob
from itertools import cycle,zip_longest
import numpy as np
import matplotlib.pyplot as plt
import imageio
import random
import os
from tensorflow.contrib import slim
import time
import sys
sys.path.insert(0, '/home/ubuntu/workspace/models/research/slim/')
from nets.resnet_v1 import resnet_v1,resnet_v1_101,resnet_arg_scope
from datasets import dataset_utils

random.seed(42)
tf.set_random_seed(42)

def grouper(n, iterable, fillvalue=None):
  args = [iter(iterable)]*n
  return zip_longest(*args, fillvalue=fillvalue)

height_image = resnet_v1.default_image_size
width_image = resnet_v1.default_image_size
print(height_image)

def get_images(data_dir, data_type, batch_size=10):
  height = height_image
  width = width_image
  if(data_type == 1):
    data_dir = data_dir +"Train/"
  elif(data_type == 2):
    data_dir = data_dir +"Val/"
  elif(data_type == 3):
    data_dir = data_dir +"Test/"
  input_files_positive = glob(data_dir + "Positive/*.jpg")
  input_files_negative = glob(data_dir + "Negative/*.jpg")
  input_files_neutral = glob(data_dir + "Neutral/*.jpg")
  #input_files_positive = glob(data_dir + "daisy/*.jpg")
  #input_files_negative = glob(data_dir + "roses/*.jpg")
  #input_files_neutral = glob(data_dir + "tulips/*.jpg")
  input_files = input_files_positive + input_files_negative + input_files_neutral
  labels = [np.array([0,0,1])]*len(input_files_positive) + [np.array([0,1,0])]*len(input_files_negative) + [np.array([1,0,0])]*len(input_files_neutral)
  label_files = list(zip(input_files,labels))
  label_files = random.sample(label_files,len(label_files))
  label_files_infinite = cycle(label_files)

  label_files_grouped = grouper(batch_size,label_files_infinite)
  while 1:
    image_names,labels = zip(*next(label_files_grouped))
    
    dx = 224
    dy = 224
    
    image_data = [Image.open(fname) for fname in image_names]
            
    images = [image.resize((256,256),PIL.Image.ANTIALIAS) if (image.size[0] < 256 or image.size[1] < 256) else image for image in image_data]
    
    images_crop = []
    for image in images:
        x = random.randint(0, image.size[0]-dx-1)
        y = random.randint(0, image.size[1]-dy-1) 
        images_crop.append(image.crop((x,y, x+dx, y+dy)))
    
    pil_images = [np.fliplr(image) if np.random.random() > 0.5 else image for image in images_crop]
    image_files = [np.array(np_image) for np_image in pil_images]
    
    yield zip(image_files,list(labels))


data_dir = "/home/ubuntu/project/data/"
total_num_images = 3630
num_epochs = 100 #hp 1
batch_size = 50 #hp 2
initial_learning_rate = 0.001 #hp 3
num_epochs_before_decay = 60
learning_rate_decay_iterations = 15 #hp 4
keep_prob_val = 0.9 #hp 5
num_batches_per_epoch = int(total_num_images / batch_size)
num_steps_per_epoch = num_batches_per_epoch 

checkpoints_dir = '/tmp/checkpoints'

tf.reset_default_graph()

images = tf.placeholder(tf.float32,shape=[None,height_image,width_image,3])
labels = tf.placeholder(tf.float32,shape=[None,3])
learning_rate = tf.placeholder(tf.float32,shape=[])
keep_prob = tf.placeholder(tf.float32,shape=[])

with slim.arg_scope(resnet_arg_scope()):
    #restore resnet101 model
    resnet_logits, end_points = resnet_v1_101(images, num_classes=1000, global_pool=True, is_training=True)
    

def feed_dict(batch_size, data_type, epoch):
  keep_prob_per = keep_prob_val
  lr = initial_learning_rate
    
  if data_type == 1:
    data = get_images(data_dir,data_type,batch_size)
    keep_prob_per = keep_prob_val

  elif data_type == 2:
    data = get_images(data_dir,data_type,batch_size)
    keep_prob_per = 1
    
  elif data_type == 3:
    data = get_images(data_dir,data_type,batch_size)
    keep_prob_per = 1
    
  imgs, lbls = map(list, zip(*next(data)))

  
  if(epoch >= num_epochs_before_decay and epoch % learning_rate_decay_iterations == 0):
    lr = 0.001*(10**(-epoch/100))

  return {images: imgs, labels: lbls, learning_rate: lr, keep_prob: keep_prob_per}

#checkpoint_exclude_scopes=["resnet_v1_101/logits"]
checkpoint_exclude_scopes=[]
exclusions = checkpoint_exclude_scopes
#review code!
variables_to_restore = []
for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
            excluded = True
            break
    if not excluded:
        variables_to_restore.append(var)
#variables_to_restore = slim.get_variables_to_restore(exclude = checkpoint_exclude_scopes)
saver = tf.train.Saver(variables_to_restore)

#introduce dropout for resnet final FC layer
logits_drop = tf.contrib.layers.dropout(end_points['global_pool'], keep_prob)
#introduce final FC layer to map output of resnet 1000 to 3 classes
logits = tf.contrib.layers.fully_connected(logits_drop, 3)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = slim.learning.create_train_op(cross_entropy, optimizer)


with tf.Session() as sess:
 
  saver.restore(sess,os.path.join(checkpoints_dir, 'resnet_v1_101.ckpt'))
  sess.run(tf.initialize_all_variables())
  
  total_loss_val = 0
  total_train_accuracy = 0
  total_val_accuracy = 0
  best_accuracy = 0
  avg_training_acc = 0
  avg_val_acc = 0
  start_time_per_epoch = 0
  time_elapsed_eval_per_epoch = 0
  time_elapsed_training_per_epoch = 0
  overall_training_time = 0
  
  overall_training_time = time.time()
  for i in range(num_epochs):
    
    best_acc_file = open('/home/ubuntu/project/L44_Mini_Project/data_dir/accuracy/best_accuracy.txt',"a") 
    all_acc_file = open('/home/ubuntu/project/L44_Mini_Project/data_dir/accuracy/all_accuracy.txt',"a") 
    
    start_time_per_epoch = time.time()
    for j in range(num_batches_per_epoch):
      print("Epoch %s, Batch %s" % (i,j))
      _,training_loss = sess.run([train_op, cross_entropy],feed_dict=feed_dict(batch_size,1,i))
      train_accuracy = accuracy.eval(feed_dict=feed_dict(batch_size,1,i))
      total_loss_val = total_loss_val + training_loss
      total_train_accuracy = total_train_accuracy + train_accuracy
    avg_training_acc = total_train_accuracy/num_batches_per_epoch
    print("--------------------------------------------------------------------")
    print("Epoch %s: training accuracy %s, training loss %s" % (i,avg_training_acc,total_loss_val))
    time_elapsed_training_per_epoch = time.time() - start_time_per_epoch
    for _ in range(num_batches_per_epoch):
      validation_accuracy = accuracy.eval(feed_dict=feed_dict(batch_size,2,i))
      total_val_accuracy = total_val_accuracy + validation_accuracy
    avg_val_acc = total_val_accuracy/num_batches_per_epoch
    print("Epoch %s: validation accuracy %s" % (i,avg_val_acc))
    print("--------------------------------------------------------------------")
    time_elapsed_eval_per_epoch = time.time() - start_time_per_epoch - time_elapsed_training_per_epoch
    all_acc_file.write("%s %s %s %s %s %s\n" % (i, total_loss_val, avg_training_acc, avg_val_acc, time_elapsed_training_per_epoch, time_elapsed_eval_per_epoch))
   
    #store model at best accuracy
    if avg_val_acc > best_accuracy:
      best_acc_file.write("Epoch %s: " % (i))
      best_acc_file.write("Best validation accuracy: %s\n" % (avg_val_acc))
      best_accuracy = avg_val_acc
      best_model_path = saver.save(sess, '/home/ubuntu/project/best_models/')  
    
    total_loss_val = 0
    total_train_accuracy = 0
    total_val_accuracy = 0
    avg_training_acc = 0
    avg_val_acc = 0
    best_model_path = ""
    start_time = 0
    time_elapsed_eval = 0
    time_elapsed_training = 0
    
    best_acc_file.close()
    all_acc_file.close()
    
  overall_training_time = time.time() - overall_training_time
  print("Training time: %s" % overall_training_time)
  
      
  #features = graph.get_tensor_by_name('inception_v3/GlobalPool')
  #features_values = sess.run(features)
  #print (features_values[0])
    