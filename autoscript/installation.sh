path_to_tensorflow=$1

#install tensorflow 1.9.0
echo "start installing tensorflow 1.9.0"
pip2 install --user tensorflow-1.9.0-cp27-cp27mu-linux_x86_64.whl

#install dependencies using pip2
echo "start installing Cython"
pip2 install --user Cython
echo "start installing contextlib2"
pip2 install --user contextlib2
echo "start installing pillow"
pip2 install --user pillow
echo "start installing lxml"
pip2 install --user lxml
echo "start installing jupyter"
pip2 install --user jupyter
echo "start installing matplotlib"
pip2 install --user matplotlib

#COCO API installation
echo "start installing pycocotools folder"
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools $path_to_tensorflow/research/

# Protobuf Compilation .From tensorflow/models/research/
echo "Protobuf Compilation"
cd $path_to_tensorflow/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.