import cv2
from absl import flags
import skimage
import os
import shutil
from scipy import *
import numpy as np
from skimage import io,color
import tensorflow as tf
from threading import Lock
from multiprocessing.pool import ThreadPool
import sys
from multiprocessing import cpu_count
import datetime

flags.DEFINE_string('data_dir', '', 'Root directory to raw NP dataset.')
flags.DEFINE_string('radius', '1', 'radius or noise level')
flags.DEFINE_string('radiusforblur', '6', 'radius or noise level')
#flags.DEFINE_integer('thread_count', 5, 'count of thread')
flags.DEFINE_string('type', '', 'blur or noise')
FLAGS = flags.FLAGS
threads = []
examples_list =[]
batch =0
thread_count=0
lock= Lock()
total_count=0
current_count=0

def thread_processFile(example):
    level = int(FLAGS.radius)
    levelforblur = int(FLAGS.radiusforblur)
    data_dir = FLAGS.data_dir
    photos_dir = os.path.join(data_dir, "photos")
    annotations_dir = os.path.join(data_dir, "photos", "Annotations")
    type = FLAGS.type

    path = os.path.join(annotations_dir, example)
    file_path = os.path.join(photos_dir, example.replace(".json", ""))
    if os.path.exists(file_path):
        if (type == "blur"):
            processFile(file_path,levelforblur)
            shutil.copyfile(path, path.replace(path[-9:], "blur" + str(levelforblur) + ".jpg.json"))
        if (type == "noise"):
            processnoiseFile(file_path, level)
            shutil.copyfile(path, path.replace(path[-9:], "noise" + str(level) + ".jpg.json"))
        if (type == "blurandnoise"):
            processFile(file_path, levelforblur)
            shutil.copyfile(path, path.replace(path[-9:], "blur" + str(levelforblur) + ".jpg.json"))
            processnoiseFile(file_path, level)
            shutil.copyfile(path, path.replace(path[-9:], "noise" + str(level) + ".jpg.json"))
    else:
        print(file_path + " not exist")


def do_process():
    percentage = -1
    data_dir = FLAGS.data_dir
    results = []
    thread_count = cpu_count()*2
    #print("cup process count is " + str(thread_count))
    pool = ThreadPool(thread_count)
    annotations_dir = os.path.join(data_dir, "photos","Annotations")
    examples_list= [f for f in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, f)) and f.find("blur")==-1 and f.find("noise")==-1]
    print("thread count is " + str(thread_count))
    #total_count=len(examples_list)
    print("total count "+str(len(examples_list)))
    for idx, example in enumerate(examples_list):
        results.append(pool.apply_async(thread_processFile, args=(example,)))
    for idx, ret in enumerate(results):
        ret.get()
        current_perc = idx * 100 // len(results)
        if current_perc != percentage:
            # print("%d%% images has been processed" % (current_perc))
            # sys.stdout.flush()
            report_progress(current_perc)
            percentage = current_perc

    job_done()
    print("no error found")

def addGaussianNoise(image,percetage):
    param = 30
    grayscale = 256
    w = image.shape[1]
    h = image.shape[0]
    newimg = np.zeros((h, w, 3), np.uint8)

    for x in range(0, h):

        for y in range(0, w, 2):
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
            z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

            fxy_0 = int(image[x, y,0] + z1)
            fxy_1 = int(image[x, y,1] + z1)
            fxy_2= int(image[x, y,2] + z1)
            if (y < w - 1):
                fxy1_0 = int(image[x, y+1, 0] + z2)
                fxy1_1 = int(image[x, y+1, 1] + z2)
                fxy1_2 = int(image[x, y+1, 2] + z2)
            # f(x,y)
            if fxy_0 < 0:
                fxy_0 = 0
            elif fxy_0 > grayscale - 1:
                fxy_0 = grayscale - 1
            else:
                fxy_0 = fxy_0

            if fxy_1 < 0:
                fxy_1 = 0
            elif fxy_1 > grayscale - 1:
                fxy_1 = grayscale - 1
            else:
                fxy_1 = fxy_1

            if fxy_2 < 0:
                fxy_2 = 0
            elif fxy_2 > grayscale - 1:
                fxy_2 = grayscale - 1
            else:
                fxy_2 = fxy_2


            if fxy1_0 < 0:
                fxy1_0 = 0
            elif fxy1_0 > grayscale - 1:
                fxy1_0 = grayscale - 1
            else:
                fxy1_0 = fxy1_0
            if fxy1_1 < 0:
                fxy1_1 = 0
            elif fxy1_1 > grayscale - 1:
                fxy1_1 = grayscale - 1
            else:
                fxy1_1 = fxy1_1
            if fxy1_2 < 0:
                fxy1_2 = 0
            elif fxy1_2 > grayscale - 1:
                fxy1_2 = grayscale - 1
            else:
                fxy1_2 = fxy1_2

            newimg[x, y, 0] = fxy_0
            newimg[x, y, 1] = fxy_1
            newimg[x, y, 2] = fxy_2
            if(y<w-1):

                newimg[x, y + 1, 0] = fxy1_0
                newimg[x, y + 1, 1] = fxy1_1
                newimg[x, y + 1, 2] = fxy1_2

    return newimg
def SaltAndPepper(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.random_integers(0,src.shape[0]-2)
        randY=random.random_integers(0,src.shape[1]-2)
        if random.random_integers(0,1)==0:

            NoiseImg[randX, randY] = 0
            NoiseImg[randX+1, randY] = 0
            NoiseImg[randX, randY+1] = 0
            NoiseImg[randX+1, randY+1] = 0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

def processFile(filepath, level):
    kernel_size = (level*2+1, level*2+1)
    sigma = 0
    img = cv2.imread(filepath)
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    new_imgName = filepath.replace(filepath[-4:],"blur"+ str(level) +".jpg")
    cv2.imwrite(new_imgName,img)

def processnoiseFile(filepath, level):
    if(filepath[-3:] == 'jpg' or filepath[-3:] == 'JPG'):
        new_imgName = filepath.replace(filepath[-4:], "noise" + str(level) + ".jpg")
    if (level & 1==1):
        img = io.imread(filepath)
        # print("noise gaussian"+filepath)
        img = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)
        io.imsave(new_imgName, img)        
    elif (level & 2==2):
        img = io.imread(filepath)
        # print("noise pepper" + filepath)
        img = skimage.util.random_noise(img, mode='pepper', seed=None, clip=True)
        io.imsave(new_imgName, img)
    elif (level & 3==3):
        img = io.imread(filepath)
        # print("noise gaussian" + filepath)
        img = skimage.util.random_noise(img, mode='salt', seed=None, clip=True)
        io.imsave(new_imgName, img)
    else:
        print('Add noise failed, please check input level')




def report_progress(percentage):
    sys.stdout.write("\r%d%% image has been processed" % percentage)

def job_done():
    report_progress(100)
    sys.stdout.write("\nDone\n")

def main(_):
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(nowTime)
    do_process()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(nowTime)

if __name__ == '__main__':
    tf.app.run()
