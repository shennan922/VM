installation description
#From tensorflow/models/research/        path_to_tensorflow=tensorflow/models
sh installation.sh /home/test_user/path_to_tensorflow

you can test if tensorflow is installed successfully using 'python2 test_tensorflow.py'

1.train_eval is the begining script.
2.+object_detection/
	train_eval.py
	blurandnoise.py
	blackout.py
	clusterring_box_for_faster_rcnn.py
	config_handle.py
	model_main_np.py
	ssd_export_opt_pb.sh
	test_json.py
	report_acc.py
	sendemail.py
  update object_detection/dataset_tools/
	create_np_tf_record.py
  update pycocotools/(if not in code find the soure update)
	cocoeval.py
3.make sure the path in ssd_export_opt_pb.sh can find graph file. 
4.the path in config needs to be modified
	fine_tune_checkpoint: "....../model.ckpt"
5.If no need to run blurandnoise.py, you don't need to transform 'radius','radiusforblur','type'
6.If you want go on train process, 'results_folder' should add the path of .../Results
7.If you don't want to split dataset, 'onedataset' set to 'Y'
8.'scales' can define by yourself according to 'num_layer'.
9.In sendemail.py, you should update the receiver email address.
10.label_map_path,pipeline_config_path,train_path will use default path, if you don't input their path.
--------------------------------------------------------------------------------------------------------------------

python2 train_eval.py \
	--data_dir=/home/test_user/dataset_harpic/rb_harpic \
	--label_map_path=/home/test_user/bin/pascal_label_map.pbtxt \
	--pipeline_config_path=/home/test_user/bin/rb_harpic_ssd_lite_mbv2_4layers.config \
	--train_path=/home/test_user/tensorflow_models/research/object_detection \
	--num_train_steps=100 \
	--num_eval_steps=10 \
	--onedataset=Y \
	--scales=[0.025,0.08,0.16,0.32] \
	--radius=2 \
	--radiusforblur=6 \
	--type=noise \
	--foldername=test \
	--results_folder=/home/test_user/dataset_harpic/Results_folder/Results20190129131726	



	//if you need go on train//--results_folder=/home/test_user/Results_folder/Results20190129131726

