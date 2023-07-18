from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import math
import tensorflow as tf
import numpy as np
import detect_face
import cv2
import time
import dlib
from math import *


def find_landmarks(src, minsize, pnet, onet, threshold, factor):
    
    img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
        
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 获取检测到的人脸的个数
    nrof_faces = bounding_boxes.shape[0]

    
    if nrof_faces == 0:
        print("no face is detected!")
        return ret, 0
    
    else:
        predictor_path = 'C:\\Users\\dwc20\\shape_predictor_68_face_landmarks.dat'
        #predictor_path = 'C:\\Users\\dwc20\\OneDrive\\Desktop\\tensorflow-mtcnn\\shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(predictor_path)
        
        # 选出最大的脸部返回
        max_face = 0
        max_face_id = 0
        id = 0
        for b in bounding_boxes:
            _area = (b[2] - b[0]) * (b[3] - b[1])
            #print("face area ", _area)
            if _area > max_face:
                max_face = _area
                max_face_id = id
                id += 1
            #cv2.rectangle(draw, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 255, 0), 2)
        
        bounding_boxes[max_face_id][0]
        rect = dlib.rectangle(int(bounding_boxes[max_face_id][0]), int(bounding_boxes[max_face_id][1]), int(bounding_boxes[max_face_id][2]), int(bounding_boxes[max_face_id][3]))
        #rect = dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        landmarks = predictor(img, rect)
        
        # 画出特征点
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # Draw a circle
            cv2.circle(img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imshow("dot", img)
        cv2.waitKey(0)
        
        # for p in points.T:
        #     for i in range(5):
        #         cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)

        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)
        return landmarks, nrof_faces

def detectBlink(src, landmarks):
    left_eye_left = landmarks.part(36)
    left_eye_right = landmarks.part(39)
    left_eye_top = landmarks.part(37)
    left_eye_buttom = landmarks.part(41)
    right_eye_left = landmarks.part(42)
    right_eye_right = landmarks.part(45)
    right_eye_top = landmarks.part(43)
    right_eye_buttom = landmarks.part(47)
    
    left_eye_height = left_eye_buttom.y - left_eye_top.y
    left_eye_width = left_eye_right.x - left_eye_left.x
    left_eye_ratio = left_eye_width / left_eye_height;
    print(left_eye_height)
    print(left_eye_width)
    print("this is ratio of width to height of left eye",left_eye_ratio)
    right_eye_height = right_eye_buttom.y - right_eye_top.y
    right_eye_width = right_eye_right.x - right_eye_left.x
    right_eye_ratio = right_eye_width / right_eye_height;
    print(right_eye_height)
    print(right_eye_width)
    print("this is ratio of width to height of right eye",right_eye_ratio)
    
    #参数待调
    if (right_eye_ratio + left_eye_ratio) / 2 > 5.5:
        return True
    else:
        return False
    x = left_eye_left.x;
    y = left_eye_left.y;
    cv2.circle(src, center = (x,y), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.imshow("src", src)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    # 准备mtcnn
    sess = tf.compat.v1.Session()
    # 创建P-Net，R-Net，O-Net网络结构，载入训练好的模型参数
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    # 最小可检测人脸图像
    minsize = 40
    # 人脸得分阈值，超过阈值，即人脸，否则，非人脸
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    # 生成图像金字塔时的缩放系数，越大阶层越大，
    factor = 0.709

    # 缩放后的尺寸minL = org_L * (12 / minisize) * factor ^ (n)，n = {0, 1, 2, 3, ...,N}，
    # 缩放尺寸最小不能小于12，也就是缩放到12为止。n的数量也就是能够缩放出图片的数量。
    
    # 获取待识别内容
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image to be detected for faces.',default='C:\\Users\\dwc20\\OneDrive\\Desktop\\tensorflow-mtcnn\\input.jpg')
    parser.add_argument('--output', type=str, help='new image with boxed faces',default='C:\\Users\\dwc20\\OneDrive\\Desktop\\tensorflow-mtcnn\\new.jpg')
    
    #获取文件类型
    argv = parser.parse_args(sys.argv[1:])
    filename = argv.input
    
    # 若为图片
    if os.path.splitext(filename)[-1] == ".jpg":
        src = cv2.imread(filename)
        landmarks, nrof_faces = find_landmarks(src, minsize, pnet, onet, threshold, factor)
        if nrof_faces > 0:
            detectBlink(src, landmarks)
        # eyeImg, monthImg = find_eye_mouth(draw, minsize, pnet, onet, threshold, factor)
        # detectBlink(eyeImg)
        # cv2.imshow("eye", eyeImg)
        # cv2.imshow("mouth", monthImg)
        cv2.waitKey(0);
    
    # 若为视频
    elif os.path.splitext(filename)[-1] == ".mp4":
        cap = cv2.VideoCapture(filename)
        while(cap.isOpened()):
            
            start = time.time();
            
            ret, src = cap.read()
            landmarks, nrof_faces = find_landmarks(src, minsize, pnet, onet, threshold, factor)
            
            if nrof_faces > 0:
                res = detectBlink(src, landmarks)
                
                if res == True:
                    cv2.putText(src, "closed", (50 , 50), cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(src, "open", (50 , 50), cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
                
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    #Draw a circle
                    cv2.circle(src, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
                
                print("\n")
                end = time.time()
                print("a frame takes",end-start)
                cv2.imshow("draw", src)
                # cv2.waitKey(0)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break