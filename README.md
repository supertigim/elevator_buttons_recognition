Introduction  
============

This project is to show how to detect and recognize buttons in an elevator for robotics.  

- Button Detection: tensorflow detection API  
- Button Recognition: OCR  
- Button Status (On/Off): Color Value Histogram   

Environment  
===========   

```
    $ conda create -n detection python=3.7 pyqt=5
    $ conda activate detection  
    (detection)$ pip install -r requirements.txt  
    (detection)$ git clone https://github.com/supertigim/elevator_buttons_recognition.git  
    (detection)$ cd elevator_buttons_recognition
    (detection)elevator_buttons_recognition$ mkdir addons && cd addons  
    (detection)elevator_buttons_recognition/addons$ git clone https://github.com/tzutalin/labelImg.git  
    (detection)elevator_buttons_recognition/addons$ cd labelImg  
    (detection)elevator_buttons_recognition/addons/labelImg$ pip install -r requirements/requirements-linux-python3.txt
    (detection)elevator_buttons_recognition/addons/labelImg$ cd ../..
    (detection)elevator_buttons_recognition$ git clone https://github.com/tensorflow/models.git

```

When labelImge doesn't work properly,  

```
    (detection)elevator_buttons_recognition/addons/labelImg$ sudo apt-get install pyqt5-dev-tools  
    (detection)elevator_buttons_recognition/addons/labelImg$ sudo pip3 install -r requirements/requirements-linux-python3.txt  
    (detection)elevator_buttons_recognition/addons/labelImg$ make qt5py3  
    pyrcc5 -o libs/resources.py resources.qrc  
```

Training  
========  

```
    # Create xmls with labelImg
    (detection)elevator_buttons_recognition$ python ./addons/labelImg/labelImg.py 

    # Convert xml to csv 
    python xml_to_csv.py -i ./images/train/ -o ./annotations/train_labels.csv
    python xml_to_csv.py -i ./images/test/ -o ./annotations/test_labels.csv

    # Convert .csv to .record
    python generate_tfrecord.py --csv_input=./annotations/train_labes.csv --output_path=./annotations/train.record --img_path=images/train/
    python generate_tfrecord.py --csv_input=./annotations/test_labes.csv --output_path=./annotations/test.record --img_path=images/test/
```

Reference  
=========  

**How to build my own button detector**  

- [Tensorflow 2 Install](https://www.tensorflow.org/install)  
- [TensorFlow Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html)  
- [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)  
- [Custom Object Detection using TensorFlow from Scratch, May 2019](https://towardsdatascience.com/custom-object-detection-using-tensorflow-from-scratch-e61da2e10087)  
- [Create your own object detector, Feb 2019](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85)  
- [TensorFlow step by step custom object detection tutorial, Jan 2019](https://medium.com/analytics-vidhya/tensorflow-step-by-step-custom-object-detection-tutorial-d7ae840a74e2)  


**Papers in regard to elevator buttons detection**  

- [Elevator Button Recognition, Tensorflow1.12 on TX2, OCR RCNN, codes based on the paper below](https://github.com/zhudelong/ocr-rcnn-v2/tree/master/src/button_recognition/scripts/ocr_rcnn_lib)  
- [A Novel OCR-RCNN for Elevator Button Recognition, 2018](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/IROS18_2028_FI.pdf)
- [Autonomous Operation of Novel Elevators for Robot Navigation, 2017](http://ai.stanford.edu/~olga/papers/icra10-OperationOfNovelElevators.pdf)

**ETC**    
 
- [Issue installing in windows10/Python 3.7/ No module named 'libs.resources'](https://github.com/tzutalin/labelImg/issues/475)  
- [Tensorflow 2.0 - AttributeError: module 'tensorflow' has no attribute 'Session'](https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session)  