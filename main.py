# -*- coding: utf-8 -*-

#!/usr/bin/env python
from __future__ import print_function
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from button_recognition import ButtonRecognition

def resize_image(image, W=480):
  height, width, depth = image.shape
  imgScale = W/width
  newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
  return cv2.resize(image,(int(newX),int(newY)))


def get_image_name_list(target_path):
    assert os.path.exists(target_path)
    image_name_list = []
    file_set = os.walk(target_path)
    for root, dirs, files in file_set:
      for image_name in files:
        if image_name.split('.')[-1] == "png": continue
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
    image_np = resize_image(image_np)
    t0 = cv2.getTickCount()
    recognizer.predict(image_np, True, True)
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


def main_video(filepath="./results/video_ex2.mov"):
  cap = cv2.VideoCapture(filepath)

  # to find the size 
  ret, frame = cap.read()
  frame = resize_image(frame)
  w = frame.shape[1]
  h = frame.shape[0]

  recognizer = ButtonRecognition()

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  #fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi
  out = cv2.VideoWriter('output.mp4',fourcc, 15.0, (int(w),int(h)))

  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret: break

      frame = resize_image(frame)
      recognizer.predict(frame, True)

      # write the flipped frame
      out.write(frame)
      # Display the resulting frame
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  #recognizer.clear_session()
  cap.release()
  out.release()
  cv2.destroyAllWindows()  

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="ElevatorButton Recognition(press 'q' for exit")
  parser.add_argument('-m', '--mode', default="cam",
                      help="Set operation mode: 'cam'-Camera Streaming, 'image'-Image Files , 'video'-Video File")
  ARGS = parser.parse_args()
  if ARGS.mode == "cam":    main_cam()
  elif ARGS.mode == "video":  main_video()
  else:                     main()

# end of file