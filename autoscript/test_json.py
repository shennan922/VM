import sys
sys.path.append('..')
import os
import json
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
import random

tf.app.flags.DEFINE_string(
    'PATH_OUTPUT', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'PATH_TO_CKPT', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'PATH_TEST_IDS', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'DIR_IMAGE', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'PATH_TO_LABELS', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'NUM_CLASSES', '', 'images directory ')
FLAGS = tf.app.flags.FLAGS

file_list = []
write_file = open(FLAGS.PATH_TEST_IDS, "w")
for file in os.listdir(FLAGS.DIR_IMAGE):
    if file.endswith(".jpg") or file.endswith(".JPG"):
      write_name = file 
      file_list.append(write_name)
      sorted(file_list)

for current_line in range(len(file_list)):
    write_file.write(file_list[current_line] + '\n')

write_file.close()

def get_results(boxes, classes, scores, category_index, im_width, im_height,
    min_score_thresh=.5):
    bboxes = list()
    for i, box in enumerate(boxes):
        if scores[i] > min_score_thresh:
          ymin, xmin, ymax, xmax = box
          bbox = {
              #{
              'x': xmin * im_width,
              'y': ymin * im_height,
              'w': (xmax-xmin) * im_width,
              'h': (ymax-ymin) * im_height,
              'id': category_index[classes[i]]['name']
              #},
                  #'category': category_index[classes[i]]['name'],
                  #'score': float(scores[i])
          }
          bboxes.append(bbox)
    return bboxes

def convert_label_map_to_categories(label_map,
                                    max_num_classes):
  categories = []
  list_of_ids_already_added = []
  if not label_map:
    label_id_offset = 1
    for class_id in range(max_num_classes):
      categories.append({
          'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)
      })
    return categories
  for item in label_map.item:
    if not 0 < item.id <= max_num_classes:
      logging.info('Ignore item %d since it falls outside of requested '
                   'label range.', item.id)
      continue
    if item.id not in list_of_ids_already_added:
      list_of_ids_already_added.append(item.id)
      categories.append({'id': item.id, 'name': item.name})
  return categories

NUM_CLASSES = FLAGS.NUM_CLASSES
label_map = label_map_util.load_labelmap(FLAGS.PATH_TO_LABELS)
categories = convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#test_ids = [line.split()[0] for line in open(FLAGS.PATH_TEST_IDS)]
test_ids = file_list
total_time = 0
test_annos_temp = dict()
flag = False

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        for image_id in test_ids:
            image_path = os.path.join(FLAGS.DIR_IMAGE, image_id)
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            im_height,im_width, _ = image_np.shape
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            start_time = time.time()
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            end_time = time.time()
            print('{} {} {:.3f}s'.format(time.ctime(), image_id, end_time - start_time))
            if flag:
                total_time += end_time - start_time
            else:
                flag = True
            test_annos_temp = get_results(
                np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index,
                im_width, im_height)
            test_annos_temp = {
              "version": "1.0.0",
              "company": "RB_Part2",
              "dataset": "Photos",
              "filename": image_id,
              "image_width": im_width,
              "image_height": im_height,
              "bndboxes" :test_annos_temp}
            fd = open(FLAGS.PATH_OUTPUT + image_id +'.json', 'w')
            json.dump(test_annos_temp, fd)
            fd.close()
print('total time: {}, total images: {}, average time: {}'.format(
    total_time, len(test_ids), total_time / len(test_ids)))
