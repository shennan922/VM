from __future__ import division
import os
import tensorflow as tf
import random
import shutil
from shutil import copy2
import re
import datetime

tf.app.flags.DEFINE_string(
    'data_dir', '../dataset_folder/dataset', 'dataset directory ')
tf.app.flags.DEFINE_string(
    'label_map_path', '', 'label_map(.pbtxt) directory.Default is ../bin/pascal_label_map.pbtxt ')
tf.app.flags.DEFINE_string(
    'pipeline_config_path', '', 'config directory.Default is ../bin/rb_harpic_ssd_lite_mbv2_4layers.config ')
tf.app.flags.DEFINE_string(
    'train_path', '', 'object_detection directory.Default is ../research/object_detection ')
tf.app.flags.DEFINE_string(
    'num_train_steps', '80000', 'Number of train steps ')
tf.app.flags.DEFINE_string(
    'num_eval_steps', '800', 'Number of eval steps ')
tf.app.flags.DEFINE_string(
    'radius', '', 'noise level .1-gaussian;2-pepper;3-salt')
tf.app.flags.DEFINE_string(
    'radiusforblur', '', 'blur level .')
tf.app.flags.DEFINE_string(
    'type', '', 'blur or noise or blurandnoise ')
tf.app.flags.DEFINE_string(
    'results_folder', '', 'result_folder directory ')
tf.app.flags.DEFINE_string(
    'scales', '[0.025,0.08,0.16,0.32]', 'scales value ')
tf.app.flags.DEFINE_string(
    'onedataset', 'Y', 'whether one dataset ')
tf.app.flags.DEFINE_string(
    'foldername', '', 'if not input detail, it will use Results+nowtime ')
FLAGS = tf.app.flags.FLAGS

def main(_):
    selfpath = os.path.dirname(os.path.realpath(__file__))

    if FLAGS.label_map_path == '':
        label_map_path = selfpath + '/../bin/pascal_label_map.pbtxt'
    else:
        label_map_path = FLAGS.label_map_path

    if FLAGS.pipeline_config_path == '':
        pipeline_config_path = selfpath + '/../bin/rb_harpic_ssd_lite_mbv2_4layers.config'
    else:
        pipeline_config_path = FLAGS.pipeline_config_path

    if FLAGS.train_path == '':
        train_path = selfpath + '/../research/object_detection'
        pwd = train_path.replace("/object_detection","")
    else:
        train_path = FLAGS.train_path
        pwd = FLAGS.train_path.replace("/object_detection","")

    os.environ['PYTHONPATH']=':'+pwd+':'+pwd+'/slim'

    if(FLAGS.results_folder == ''):
        #create train,eval folders
        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        folder_dir = FLAGS.data_dir + '/../Results_folder'
        if FLAGS.foldername == '':
            results_folder = folder_dir + '/Results'+now_time
        else:
            results_folder = folder_dir + '/Results_'+FLAGS.foldername

        if os.path.exists(folder_dir) ==False:
            os.mkdir(folder_dir)
        if os.path.exists(results_folder) ==False:
            os.mkdir(results_folder)
        if os.path.exists(results_folder+'/image_eval') ==False:
            os.mkdir(results_folder+'/image_eval')
        if os.path.exists(results_folder+'/image_train') ==False:
            os.mkdir(results_folder+'/image_train')
        if os.path.exists(results_folder+'/image_eval/photos') ==False:
            os.mkdir(results_folder+'/image_eval/photos')
        if os.path.exists(results_folder+'/image_train/photos') ==False:
            os.mkdir(results_folder+'/image_train/photos')
        if os.path.exists(results_folder+'/TFRecord_eval') ==False:
            os.mkdir(results_folder+'/TFRecord_eval')
        if os.path.exists(results_folder+'/TFRecord_train') ==False:
            os.mkdir(results_folder+'/TFRecord_train')
        if os.path.exists(results_folder+'/pb') ==False:
            os.mkdir(results_folder+'/pb')
        if os.path.exists(results_folder+'/model') ==False:
            os.mkdir(results_folder+'/model')
        if os.path.exists(results_folder+'/image_eval/photos/AnnotationsPred') ==False:
            os.mkdir(results_folder+'/image_eval/photos/AnnotationsPred')
        evalDir=results_folder+'/image_eval'
        trainDir=results_folder+'/image_train'

        trainfiles = os.listdir(FLAGS.data_dir + '/photos')
        file_list = []
        for filename in trainfiles:
            if filename.endswith('jpg') or filename.endswith('JPG'):
                write_name = filename
                file_list.append(write_name)
        index_list = list(range(len(trainfiles)))
        random.shuffle(index_list)
        num = 1
        num_train = len(file_list)

        if FLAGS.onedataset == 'Y':
            for i in index_list:
                fileName = os.path.join((FLAGS.data_dir + '/photos'), trainfiles[i])
                if(trainfiles[i][-3:] == 'jpg' or trainfiles[i][-3:] == 'JPG'):
                    copy2(fileName, (trainDir + '/photos'))
                    copy2(fileName, (evalDir + '/photos'))
        else:
            for i in index_list:
                fileName = os.path.join((FLAGS.data_dir + '/photos'), trainfiles[i])
                if(trainfiles[i][-3:] == 'jpg' or trainfiles[i][-3:] == 'JPG'):
                    n=num/num_train
                    if n <= 0.1:
                        copy2(fileName, (evalDir + '/photos'))
                    else:
                        copy2(fileName, (trainDir + '/photos'))
                    num += 1
        copy2((FLAGS.data_dir + '/templates.json'),evalDir)
        copy2((FLAGS.data_dir + '/templates.json'),trainDir)
        #---------------------------------------------------------------------------------------------------
        #create Annotations folder
        if os.path.exists(trainDir + '/photos/Annotations') ==False:
            os.mkdir(trainDir + '/photos/Annotations')
            print('Create train Annotations folder success')
        if os.path.exists(evalDir + '/photos/Annotations') ==False:
            os.mkdir(evalDir + '/photos/Annotations')
            print('Create eval Annotations folder success')
        #---------------------------------------------------------------------------------------------------
        train_file_list = []

        write_file_name = trainDir + '/train.txt'
        write_file = open(write_file_name, "w")
        for file in os.listdir(trainDir + '/photos'):
            if file.endswith(".jpg") or file.endswith(".JPG"):
              write_name = file
              train_file_list.append(write_name)      
              sorted(file_list)

        for current_line in range(len(train_file_list)):
            write_file.write(train_file_list[current_line] + '.json' + '\n')

        write_file.close()
        count=0
        with open(write_file_name,'r')as f:
                for line in f:
                    line=line.split('\n')
                    count=count+1
                    copy2(FLAGS.data_dir + '/photos/Annotations/' + line[0],trainDir + '/photos/Annotations')
        print('Train json files end')

        eval_file_list = []
        write_file_name = evalDir + '/eval.txt'
        write_file = open(write_file_name, "w")
        for file in os.listdir(evalDir + '/photos'):
            if file.endswith(".jpg") or file.endswith(".JPG"): 
              write_name = file
              eval_file_list.append(write_name)
              sorted(file_list)

        for current_line in range(len(eval_file_list)):
            write_file.write(eval_file_list[current_line] + '.json' + '\n')

        write_file.close()
        count=0
        with open(write_file_name,'r')as f:
                for line in f:
                    line=line.split('\n')
                    count=count+1
                    copy2(FLAGS.data_dir + '/photos/Annotations/' + line[0],evalDir + '/photos/Annotations')
        print('Eval json files end')

        if FLAGS.onedataset == 'Y':
            #blackoutforeval
                returnvalue=os.system("python2 blackout.py --input_folder=" + results_folder +"/image_eval --out_folder=" + results_folder +"/image_eval")
                if returnvalue == 0:
                    print('evaldataset blackout success!')
                else:
                    print('When blackout ,error occurred , stop next step ,please check your files.')
                    exit()
        #---------------------------------------------------------------------------------------------------
        #blurornoise
        if(FLAGS.type != ''):
            returnvalue=os.system("python2 blurandnoise.py\
                --data_dir=" + results_folder +"/image_train\
                --type=" + FLAGS.type +" \
                --radiusforblur=" + FLAGS.radiusforblur +" \
                --radius=" + FLAGS.radius +"")
            if returnvalue == 0:
                print('blur or noise success!')
            else:
                print('When blurornoise ,Eerror occurred , stop next step ,please check your files.')
                exit()
        else:
            print('No need to execute blur and noise.')
        #---------------------------------------------------------------------------------------------------
        #blackout
        returnvalue=os.system("python2 blackout.py --input_folder=" + results_folder +"/image_train --out_folder=" + results_folder +"/image_train")
        if returnvalue == 0:
            print('traindataset blackout success!')
        else:
            print('When blackout ,error occurred , stop next step ,please check your files.')
            exit()
        #---------------------------------------------------------------------------------------------------
        #TF_Record
        if FLAGS.onedataset == 'Y':
            returnvalue=os.system("python2 " + train_path +"/dataset_tools/create_np_tf_record.py\
            --label_map_path=" + label_map_path +"\
            --data_dir=" + trainDir +"\
            --output_path=" + results_folder +"/TFRecord_train/np_train_onedataset.record-00000-of-00010")
            if returnvalue == 0:
                print('onedataset train TFRecord success!')
            else:
                print('When make train TFRecord ,error occurred , stop next step ,please check your files.')
                exit()
        else:
            returnvalue=os.system("python2 " + train_path +"/dataset_tools/create_np_tf_record.py\
                --label_map_path=" + label_map_path +"\
                --data_dir=" + trainDir +"\
                --output_path=" + results_folder +"/TFRecord_train/np_train.record-00000-of-00010")
            if returnvalue == 0:
                print('train TFRecord success!')
            else:
                print('When make train TFRecord ,error occurred , stop next step ,please check your files.')
                exit()
            returnvalue=os.system("python2 " + train_path +"/dataset_tools/create_np_tf_record.py\
                --label_map_path=" + label_map_path +"\
                --data_dir=" + evalDir +"\
                --output_path=" + results_folder +"/TFRecord_eval/np_eval.record-00000-of-00010")
            if returnvalue == 0:
                print('eval TFRecord success!')
            else:
                print('When make eval TFRecord ,error occurred , stop next step ,please check your files.')
                exit()  
        #---------------------------------------------------------------------------------------------------
        #scale ratio
        count_jpgfile = 0
        ls = os.listdir(FLAGS.data_dir)
        for i in ls:
            if i.endswith(".jpg") or i.endswith(".JPG"):
                count_jpgfile += 1
        NUM_CLASSES = count_jpgfile

        os.system("python2 clusterring_box_for_faster_rcnn.py --input_folder=" + results_folder +"/image_train")

        if FLAGS.onedataset == 'Y':
            returnvalue=os.system("python2 config_handle.py\
            --num_classes=" + str(NUM_CLASSES) +" \
            --pipeline_config_path=" + pipeline_config_path +" \
            --pipeline_config_path_new=" + results_folder + "/image_train/new.config \
            --scales=" + FLAGS.scales +" \
            --train_path=" + results_folder +"/TFRecord_train/np_train_onedataset.record-00000-of-00010 \
            --eval_path=" + results_folder +"/TFRecord_train/np_train_onedataset.record-00000-of-00010 \
            --label_map_path=" + label_map_path +" \
            --input_folder=" + results_folder +"/image_train")
            if returnvalue == 0:
                print('onedataset pipeline_config update success!')
            else:
                print('When pipeline_config updating ,error occurred , stop next step ,please check your files.')
                exit()
        else:
            returnvalue=os.system("python2 config_handle.py\
                --num_classes=" + str(NUM_CLASSES) +" \
                --pipeline_config_path=" + pipeline_config_path +" \
                --pipeline_config_path_new=" + results_folder + "/image_train/new.config \
                --scales=" + FLAGS.scales +" \
                --train_path=" + results_folder +"/TFRecord_train/np_train.record-00000-of-00010 \
                --eval_path=" + results_folder +"/TFRecord_eval/np_eval.record-00000-of-00010 \
                --label_map_path=" + label_map_path +" \
                --input_folder=" + results_folder +"/image_train")
            if returnvalue == 0:
                print('pipeline_config update success!')
            else:
                print('When pipeline_config updating ,error occurred , stop next step ,please check your files.')
                exit()
    else:
        results_folder = FLAGS.results_folder
    #---------------------------------------------------------------------------------------------------
    #Auto train
    returnvalue=os.system("python2 " + train_path +"/model_main_np.py\
        --pipeline_config_path=" + results_folder + "/image_train/new.config \
        --model_dir=" + results_folder + "/model \
        --num_train_steps=" + FLAGS.num_train_steps +" \
        --num_eval_steps=" + FLAGS.num_eval_steps +"")
    if returnvalue == 0:
        print('Train success!')
    else:
        print('When training ,error occurred , stop next step ,please check your files.')
        exit()
    #---------------------------------------------------------------------------------------------------
    #create pb file
    filelist = []
    pattern = re.compile(r'\d+')
    for lists in os.listdir(results_folder + "/model"):
        if lists[:10]=='model.ckpt' and lists[-4:]=='meta':
            res = re.findall(pattern, lists)
            writename = res
            filelist.append(writename)
    filelist = str(filelist).replace("['","").replace(" ","").replace("']","").replace("[","").replace("]","").replace("\n","").split(',')
    filelist = map(int,filelist)
    lastckpt = 'model.ckpt-' + str(max(filelist))
    print(lastckpt)

    returnvalue=os.system("python2 " + train_path +"/export_inference_graph.py \
        --pipeline_config_path=" + results_folder + "/image_train/new.config \
        --trained_checkpoint_prefix=" + results_folder +"/model/" + lastckpt +" \
        --output_directory=" + results_folder +"/pb")
    if returnvalue == 0:
        print('Create pb file success!')
    else:
        print('When creating pb file ,error occurred , stop next step ,please check your files.')
        exit()
    #---------------------------------------------------------------------------------------------------
    #process pb
    returnvalue=os.system("sh ssd_export_opt_pb.sh " + results_folder +"")
    if returnvalue == 0:
        print('Process pb file success!')
    else:
        print('When processing pb file ,error occurred , stop next step ,please check your files.')
        exit()
    #---------------------------------------------------------------------------------------------------
    returnvalue=os.system("python2 test_json.py \
        --PATH_OUTPUT=" + results_folder +"/image_eval/photos/AnnotationsPred/ \
        --PATH_TO_CKPT=" + results_folder +"/pb/opt.pb \
        --PATH_TEST_IDS=" + results_folder +"/image_eval/photos/jpg.txt \
        --DIR_IMAGE=" + results_folder +"/image_eval/photos \
        --PATH_TO_LABELS=" + label_map_path +" \
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
        --path=" + results_folder +"/image_eval/result.txt \
        --create_pb_step=" + FLAGS.num_train_steps + " \
        --data_dir=" + FLAGS.data_dir + " \
        --results_folder=" + results_folder +"")
    print('Send accuracy report end')
if __name__ == '__main__':
    tf.app.run()
