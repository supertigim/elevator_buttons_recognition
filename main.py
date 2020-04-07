# -*- coding: utf-8 -*-

#!/usr/bin/env python
from __future__ import print_function
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from button_recognition import ButtonRecognition

def get_image_name_list(target_path):
    assert os.path.exists(target_path)
    image_name_list = []
    file_set = os.walk(target_path)
    for root, dirs, files in file_set:
      for image_name in files:
        image_name_list.append(image_name.split('.')[0])
    return image_name_list

def main():
  data_dir = './test_panels'
  data_list = get_image_name_list(data_dir)
  recognizer = ButtonRecognition()
  overall_time = 0

  for data in data_list:
    img_path = os.path.join(data_dir, data+'.jpg')
    image_np = np.copy(np.asarray(Image.open(tf.gfile.GFile(img_path, 'rb'))))
    t0 = cv2.getTickCount()
    recognizer.predict(image_np, True)
    t1 = cv2.getTickCount()
    
    time = (t1-t0)/cv2.getTickFrequency()
    overall_time += time
    print('Time elapsed: {}'.format(time))

    image = Image.fromarray(image_np)
    image.show()  

  average_time = overall_time / len(data_list)
  print('Average_used:{}'.format(average_time))
  recognizer.clear_session()


def main_cam():
  cap = cv2.VideoCapture(0)
  recognizer = ButtonRecognition()
  
  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      recognizer.predict(frame, True)
      # Display the resulting frame
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  recognizer.clear_session()
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="ElevatorButton Recognition(press 'q' for exit")
  parser.add_argument('-m', '--mode', default="cam",
                      help="Set operation mode: 'cam'-Camera Streaming, 'file'-Images from Directory")
  ARGS = parser.parse_args()
  if ARGS.mode == "cam":  main_cam()
  else:                   main()

# end of file