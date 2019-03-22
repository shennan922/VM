from __future__ import division
import os
import tensorflow as tf
import random
import shutil
from shutil import copy2
import re
import datetime

tf.app.flags.DEFINE_string(
    'data_dir', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'label_map_path', '', 'label_map directory ')
tf.app.flags.DEFINE_string(
    'pipeline_config_path', '', 'pipeline_config directory ')
tf.app.flags.DEFINE_string(
    'train_path', '', 'train directory ')
tf.app.flags.DEFINE_string(
    'num_train_steps', '', 'Number of train steps ')
tf.app.flags.DEFINE_string(
    'num_eval_steps', '', 'Number of eval steps ')
tf.app.flags.DEFINE_string(
    'radius', '', 'noise level .1-gaussian;2-pepper;3-salt')
tf.app.flags.DEFINE_string(
    'radiusforblur', '', 'radius level .')
tf.app.flags.DEFINE_string(
    'type', '', 'blur or noise ')
tf.app.flags.DEFINE_string(
    'results_folder', '', 'result_folder directory ')
tf.app.flags.DEFINE_string(
    'scales', '', 'scales value ')
tf.app.flags.DEFINE_string(
    'onedataset', '', 'whether distingush dataset ')
tf.app.flags.DEFINE_string(
    'foldername', '', 'foldername ')
FLAGS = tf.app.flags.FLAGS

train_path = FLAGS.train_path.replace("/object_detection","")
os.environ['PYTHONPATH']=':'+train_path+':'+train_path+'/slim'


results_folder = FLAGS.results_folder

count_jpgfile = 0
ls = os.listdir(FLAGS.data_dir)
for i in ls:
    if i.endswith(".jpg") or i.endswith(".JPG"):
        count_jpgfile += 1
NUM_CLASSES = count_jpgfile
#---------------------------------------------------------------------------------------------------
returnvalue=os.system("python2 test_json.py \
    --PATH_OUTPUT=" + results_folder +"/image_eval/photos/AnnotationsPred/ \
    --PATH_TO_CKPT=" + results_folder +"/pb/opt.pb \
    --PATH_TEST_IDS=" + results_folder +"/image_eval/photos/jpg.txt \
    --DIR_IMAGE=" + results_folder +"/image_eval/photos \
    --PATH_TO_LABELS=" + FLAGS.label_map_path +" \
    --NUM_CLASSES="'NUM_CLASSES'"")
if returnvalue == 0:
    print('Evaluate success!')
else:
    print('When evaluating ,error occurred , stop next step ,please check your files.')
    exit()

returnvalue=os.system("python2 report_acc.py -pj=" + results_folder +"/image_eval")
if returnvalue == 0:
    print('Generate accuracy report success!')
else:
    print('When generating accuracy report ,error occurred , stop next step ,please check your files.')
    exit()
#---------------------------------------------------------------------------------------------------
#send email
os.system("python2 sendemail.py \
    --path=" + results_folder +"/image_eval/result.txt")
print('Send accuracy report end')
