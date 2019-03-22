import argparse
import cv2
import json
from shutil import copytree
import os
import sys
import fnmatch
from os.path import abspath, join, splitext, basename
import numpy as np
from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder",required=True, type=str, help="put entire directory")
args = parser.parse_args()

DEBUG = False

def find_image_files(out_folder):
    photos_path = join(out_folder, 'photos')
    if not os.path.exists(photos_path):
        photos_path = join(out_folder, 'Photos')
    assert os.path.exists(photos_path), "photo folder is not found at {}".format(photos_path)
    all_files = os.listdir(photos_path)
    photo_files = []
    for file in all_files:
        if not file.startswith("."):
            if    fnmatch.fnmatch(file, '*.jpg') \
               or fnmatch.fnmatch(file, '*.JPG'):
                photo_files.append(os.path.join(photos_path, file))
    photo_files = sorted(photo_files)
    return photo_files, photos_path


def find_annotation_files(photos_path):
    annotation_path = join(photos_path, "Annotations")
    assert os.path.exists(annotation_path), "Annotation folder is not found at {}".format(annotation_path)
    all_files = os.listdir(annotation_path)
    annotation_files = []
    for file in all_files:
        if not file.startswith("."):
            if fnmatch.fnmatch(file, '*.json') or fnmatch.fnmatch(file, '*.JSON'):
                annotation_files.append(os.path.join(annotation_path, file))
    annotation_files = sorted(annotation_files)
    return annotation_files


def find_files(out_folder):
    photo_files, photos_path = find_image_files(out_folder)
    annotation_files = find_annotation_files(photos_path)

    #assert len(photo_files) == len(annotation_files), "Number of images and annotations are not the same!"
    files = [(c, b) for c, b in zip(photo_files, annotation_files)]

    def check_files(photo_file, annotation_file):
        image_file = splitext(basename(annotation_file))[0]
        photo = basename(photo_file)
        assert image_file == photo, "Mismatch in image:%s and annotations:%s" % (image_file, photo)

    for (photo_file, annotation_file) in files:
        check_files(photo_file, annotation_file)
    return files





def resize_json_image(json_data, image, json_path, photo_file):
    new_w, new_h = target_w, target_h
    (h, w, c) = image.shape
    bndboxes = json_data["bndboxes"]
    if len(bndboxes) == 0:
        pass
    else:
        xmin, xmax, ymin, ymax, area = 10000, 0, 10000, 0, []  # arbitrary
        for bndboxes in json_data["bndboxes"]:
            xmin = min(xmin, bndboxes["x"])
            ymin = min(ymin, bndboxes["y"])
            xmax = max(xmax, (bndboxes["x"] + bndboxes["w"]))
            ymax = max(ymax, (bndboxes["y"] + bndboxes["h"]))
            if xmax > w or ymax > h:
                print("There is a bndbox outside of height and width")
                print("The filename is : " + json_data["filename"])
                print("xmax: {} , width : {}".format(xmax, w))
                print("ymax: {} , height : {}".format(ymax, h))
                print("The bndbox is : {} ".format(bndboxes))
                exit()
            area.append(bndboxes["w"] * bndboxes["h"])

        area.sort()
        idx = int(len(area) * .1)
        if idx == 0:
            idx = 1

        areas_smallest_10percent = area[:idx]
        area_min = np.mean(areas_smallest_10percent)
        k_min_pixel = min(np.sqrt(float(args.min_pixel) / area_min), 1)
        new_h = h * k_min_pixel
        if new_h < target_h:
            new_h = target_h

    new_w = new_h * 1.0 / h * w
    if new_w < target_w:
        new_w = target_w
        new_h = int(new_w * 1.0 / w * h)

    k_resize = min(1.0, new_h * 1.0 / h)
    new_w = int(w * k_resize)
    new_h = int(h * k_resize)
    # json_data["image_width"] = new_w
    # json_data["image_height"] = new_h
    for i in range(len(json_data["bndboxes"])):
        json_data["bndboxes"][i]["x"] = int(json_data["bndboxes"][i]["x"] * k_resize)
        json_data["bndboxes"][i]["y"] = int(json_data["bndboxes"][i]["y"] * k_resize)
        json_data["bndboxes"][i]["w"] = int(json_data["bndboxes"][i]["w"] * k_resize)
        json_data["bndboxes"][i]["h"] = int(json_data["bndboxes"][i]["h"] * k_resize)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    aspect_ratio = new_w * 1.0 / new_h

    if aspect_ratio - 0.75 > 0.05:
        # too fat
        new_h2 = int(new_w * 4 / 3)
        image = cv2.copyMakeBorder(image, 0, new_h2 - new_h, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        new_h = new_h2

    if aspect_ratio - 0.75 < - 0.05:
        # too narrow
        new_w2 = int(new_h * 3 / 4)
        image = cv2.copyMakeBorder(image, 0, 0, 0, new_w2 - new_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        new_w = new_w2

    json_data["image_width"] = new_w
    json_data["image_height"] = new_h

    WRITE = True
    if WRITE:
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        cv2.imwrite(photo_file, image)


def change_to_wh(json_data, image):
    bndboxes = json_data.get("bndboxes")

    #ignore_list = list(filter(lambda box: "ignore" in box and box["ignore"], bndboxes))
    #for obj in ignore_list:
    #    box = obj
    #    w = int(box["w"])
    #    h = int(box["h"])
    #    x = int(box["x"])
    #    y = int(box["y"])
    #    image[y:y + h, x:x + w, :] = 0

    # if dirty:
    # print("Saving blackout file " + full_path)
    # cv2.imwrite(full_path+".black_out.jpg", image)
    #    cv2.imwrite(full_path, image)

    (h, w, c) = image.shape
    bndboxes = list(filter(lambda box: "ignore" not in box or not box["ignore"], bndboxes))
    size_w = []
    size_h = []
    for box in bndboxes:
        d_w = int(box["w"])*1.0/w
        d_h = int(box["h"])*1.0/w
        #x = int(box["x"])
        #y = int(box["y"])
        size_w.append(d_w)
        size_h.append(d_h)

    return size_w, size_h
    # return image

target_w = 600

from sklearn.cluster import KMeans
from sklearn.neighbors.kde import KernelDensity
if __name__ == '__main__':
    file_list = []
    write_file = open(args.input_folder + '/scales.txt', "w")
    
    input_folder = abspath(args.input_folder)
    # find the photo files and annotation files
    files = find_files(input_folder)
    total = len(files)


    def worker(photo_path, json_path):
        try:
            image = cv2.imread(photo_path)
        except:
            print("can't open jpg file:%s" % (photo_path))

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        return change_to_wh(json_data,image)


    results = []
    pool = ThreadPool(8)
    counter = 0
    print("Start iterating over files. Total files: {}".format(total))
    for (photo_path, json_path) in files:
        results.append(pool.apply_async(worker, args=("" + photo_path, json_path)))

        # if counter % 20 == 0:
        #    print("{}% Completed: {}/{}".format(done, counter, total))
    percentage = -1

    all_w=[]
    all_h=[]
    for idx, ret in enumerate(results):
        w,h=ret.get()
        all_w=all_w+w
        all_h=all_h+h
        #current_perc = idx * 100 // len(results)
        #if current_perc != percentage:
        #    sys.stdout.write("\r%d%% images has been processed" % (current_perc))
        #    sys.stdout.flush()
        #    percentage = current_perc

    np_w=np.array(all_w)
    np_h=np.array(all_h)

    np_aspect = np_w/np_h



    K = KMeans(3, random_state=1)
    np_aspect_2d=np.repeat(np_aspect[:, np.newaxis], 2, axis=1)
    centers=K.fit(np_aspect_2d).cluster_centers_
    print("\n")
    print("aspect:[", centers[0, 0], centers[1,0], centers[2,0],"]")
    aspect_list=[centers[0, 0], centers[1,0], centers[2,0]]
    write_file.write(str(aspect_list))
    write_file.write('\n')

    np_wh = np.stack((np_w,np_h), 1)*600
    np_wh2 = np_wh*1.5
    #np_wh3 = np_wh*3
    #np_wh_all=np.concatenate((np_wh,np_wh2, np_wh3))
    np_wh_all=np.concatenate((np_wh,np_wh2))
    K = KMeans(3, random_state=1)
    centers=K.fit(np_wh_all).cluster_centers_

    #kde = KernelDensity().fit(np_aspect)
    print("scale", np.sqrt(centers[:,0]*centers[:,1])/256)
    scales_list=str(np.sqrt(centers[:,0]*centers[:,1])/256)
    write_file.write(str(scales_list))
    print("\nEnd")
    #print(aspect_list)
    #print(scales_list)
    write_file.close()
