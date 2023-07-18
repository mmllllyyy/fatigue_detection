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
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
# from keras.engine.saving import load_model
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# 获取眼部图像标签
def get_eye_label(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = eye_model.predict(img)
    i = preds.argmax(axis=1)[0]
    if i > 0:
        label = "open_eye"
    else :
        label = "closed_eye"

    return label

    return label

# 获取嘴部图像标签
def get_mouth_label(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = mouth_model.predict(img)
    i = preds.argmax(axis=1)[0]
    if i > 0:
        label = "open_mouth"
    else :
        label = "closed_mouth"

    return label

# 裁剪眼部
def crop_eyes(img, points, face_id):
    # 左上角坐标(xl,yl)，右下角坐标(xr,yr)，右上角坐标（xru,yru），左下角(xld,yld)
    # 左眼中心坐标(x1,y1)，右眼中心坐标(x2,y2)
    xl = 0
    yl = 0
    xr = 0
    yr = 0
    xld = 0
    yld = 0
    xru = 0
    yru = 0
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x3=0
    y3=0
    x4=0
    y4=0

    p = points.T[face_id]
    for i in range(5):
        # cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
        # 左眼中心点位置
        if i == 0:
            # cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 255, 0), 2)
            x1 = p[i]
            y1 = p[i + 5]
        # 右眼中心点位置
        if i == 1:
            # cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 255, 0), 2)
            x2 = p[i]
            y2 = p[i + 5]
        # 左嘴角中心点位置
        if i == 3:
            # cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 255, 0), 2)
            x3 = p[i]
            y3 = p[i + 5]
        # 右嘴角中心点位置
        if i == 4:
            # cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 255, 0), 2)
            x4 = p[i]
            y4 = p[i + 5]

    # 左右眼中心点坐标，旋转得到变换后坐标
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    # 左右嘴角点坐标，旋转得到变换后坐标
    pt3 = (int(x3), int(y3))
    pt4 = (int(x4), int(y4))

    # imgRotation, eye_left_point, eye_right_point, mouth_left_point, mouth_right_point = rotate(img, pt1, pt2, pt3, pt4)
    eye_left_point = pt1
    eye_right_point = pt2
    mouth_left_point = pt3
    mouth_right_point = pt4
    imgRotation = img
    eye_left_point = (int(eye_left_point[0]), int(eye_left_point[1]))
    eye_right_point = (int(eye_right_point[0]), int(eye_right_point[1]))
    # 两眼间距
    d = math.sqrt((eye_right_point[1] - eye_left_point[1]) * (eye_right_point[1] - eye_left_point[1]) + (eye_right_point[0] - eye_left_point[0]) * (eye_right_point[0] - eye_left_point[0]))
    d_avg = int(d/3)

    # 左眼和右眼区域图像
    eye_left_region = imgRotation[int(eye_left_point[1]-0.75*d_avg):int(eye_left_point[1]+0.75*d_avg),
                        int(eye_left_point[0]-d_avg):int(eye_left_point[0]+d_avg)]
    eye_right_region = imgRotation[int(eye_right_point[1] - 0.75 * d_avg):int(eye_right_point[1] + 0.75 * d_avg),
                        int(eye_right_point[0] - d_avg):int(eye_right_point[0] + d_avg)]

    # 左右眼区域拼接，横向连接
    # image = np.concatenate([eye_left_region, eye_right_region], axis=1)
    
    # 截取嘴巴区域图像
    mouth_center_x = (mouth_left_point[0]+mouth_right_point[0])/2
    mouth_center_y = (mouth_left_point[1]+mouth_right_point[1])/2

    mouth_region = imgRotation[int(mouth_center_y-d_avg):int(mouth_center_y+d_avg),
                        int(mouth_center_x-2*d_avg):int(mouth_center_x+2*d_avg)]



    # cv2.imshow("imgOut1", cv2.cvtColor(eye_left_region, cv2.COLOR_BGR2RGB))
    # cv2.imshow("imgOut2", cv2.cvtColor(eye_right_region, cv2.COLOR_BGR2RGB))
    # cv2.imshow("imgOut3", cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB))
    
    # cv2.moveWindow('imgOut1', 0, 0)
    # cv2.moveWindow('imgOut2', eye_left_region.shape[1], 0)
    # cv2.moveWindow('imgOut3', eye_left_region.shape[1] * 2, 0)
    # cv2.waitKey(0)
    return eye_left_region, eye_right_region, mouth_region

# MTCNN
def MTCNN(img, minsize, pnet, rnet, onet, threshold, factor):
    #人脸检测
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    return bounding_boxes, points

# 获取识别到的最大脸部id
def get_face_id(bounding_boxes):
    max_face_id = 0
    max_face_area = 0
    id = 0
    for b in bounding_boxes:
        _area = (b[2] - b[0]) * (b[3] - b[1])
        #print("face area ", _area)
        if _area > max_face_area:
            max_face_area = _area
            max_face_id = id
            id += 1
        #cv2.rectangle(draw, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 255, 0), 2)
    return max_face_id


# main
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
        
    # 准备CNN    
    eye_model_path = "C:\\Users\\mmllllyyy\\OneDrive\\Desktop\\face_detect\\final_eye.h5"
    mouth_model_path = "C:\\Users\\mmllllyyy\\OneDrive\\Desktop\\face_detect\\final_mouth.h5"
    # eye_model_path = "C:\\Users\\dwc20\\OneDrive\\Desktop\\face_detect\\final_eye.h5"
    # mouth_model_path = "C:\\Users\\dwc20\\OneDrive\\Desktop\\face_detect\\final_mouth.h5"
    eye_model = load_model(eye_model_path, compile=False)
    mouth_model = load_model(mouth_model_path, compile=False)
    
    
    # 获取待识别内容
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image to be detected for faces.',default='C:\\Users\\dwc20\\OneDrive\\Desktop\\tensorflow-mtcnn\\input.jpg')
    parser.add_argument('--output', type=str, help='new image with boxed faces',default='C:\\Users\\dwc20\\OneDrive\\Desktop\\tensorflow-mtcnn\\new.jpg')
    
    # 获取文件类型
    argv = parser.parse_args(sys.argv[1:])
    filename = argv.input
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 720, 1280)
    
    # 若为图片
    if os.path.splitext(filename)[-1] == ".jpg":
        src = cv2.imread(filename)
        draw = src
        img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
        
        bounding_boxes, points = MTCNN(img, minsize, pnet, rnet, onet, threshold, factor)
        
        # 获取检测到的人脸的个数
        nrof_faces = bounding_boxes.shape[0]
        
        # 选出最大的脸部
        max_face_id = get_face_id(bounding_boxes)
        
        # 画出脸部框
        b = bounding_boxes[max_face_id]
        cv2.rectangle(draw, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 255, 0), 2)
        
        #画出五个关键点
        # for p in points.T:
        #     for i in range(5):
        #         cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)
                    
        
        
        if nrof_faces > 0:
            # 截取眼嘴图像
            eye_left_region, eye_right_region, mouth_region = crop_eyes(img, points, max_face_id)
            
            # 判断左眼状态
            if get_eye_label(eye_left_region)=='closed_eye':
                left_state='closed'
            else:
                left_state='open'
            
            # 左眼状态画图    
            left_eye_x = int(points.T[max_face_id][0])
            left_eye_y = int(points.T[max_face_id][5])
            
            cv2.putText(draw, "{}".format(left_state), (left_eye_x-30, left_eye_y-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),1, 8)
            
            # 判断右眼状态
            if get_eye_label(eye_right_region) == 'closed_eye':
                right_state='close'
            else:
                right_state='open'
            
            # 右眼状态画图
            right_eye_x = int(points.T[max_face_id][1])
            right_eye_y = int(points.T[max_face_id][6])
            
            cv2.putText(draw, "{}".format(right_state), (right_eye_x-50, right_eye_y+40), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 0), 1, 8)
            
            # 判断嘴部状态
            mouth_label = get_mouth_label(mouth_region)
            if mouth_label == 'closed_mouth':
                mouth_state='close'
            elif mouth_label == 'open_mouth':
                mouth_state='open'
            else:
                mouth_state='close'
                
            # 嘴部状态画图
            mouth_x = int((points.T[max_face_id][4]-points.T[max_face_id][3])/2+points.T[max_face_id][3])
            mouth_y = int((points.T[max_face_id][9] + points.T[max_face_id][8])/2)
            cv2.putText(draw, "{}".format(mouth_state), (mouth_x-80, mouth_y), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 1, 8)
            
            cv2.imshow("img", draw)
            cv2.waitKey(0)
            
    # 若为视频
    times=[]
    if os.path.splitext(filename)[-1] == ".mp4":
        cap = cv2.VideoCapture(filename)
        
        fatigue = 0
        
        # 初始化计数器
        total_frames = 0
        closed_frames = 0
        
        # 设定计算 PERCLOS 的时间窗口（单位：秒）
        time_window = 3
        # 获取视频的帧数（每秒有多少帧）
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算所需的帧数（将时间窗口长度乘以每秒帧数）
        frame_window = int(time_window * fps)
        
        # 嘴部状态数据列表
        mouth_states = []  
        
        while(cap.isOpened()):
            ret, src = cap.read()
            start = time.time()
            
            if not ret:
                print("已到达视频末尾")
                break
            
            draw = src
            img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
            
            total_frames += 1
            
            bounding_boxes, points = MTCNN(img, minsize, pnet, rnet, onet, threshold, factor)
            
            # 获取检测到的人脸的个数
            nrof_faces = bounding_boxes.shape[0]
            
            # 选出最大的脸部
            max_face_id = get_face_id(bounding_boxes)
            
            # 画出脸部框
            b = bounding_boxes[max_face_id]
            cv2.rectangle(draw, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 255, 0), 2)
            
            #画出五个关键点
            for p in points.T:
                for i in range(5):
                    cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)   
            
            
            if nrof_faces > 0:
                # 截取眼嘴图像
                eye_left_region, eye_right_region, mouth_region = crop_eyes(img, points, max_face_id)
                
                # 判断左眼状态
                if get_eye_label(eye_left_region)=='closed_eye':
                    left_state='closed'
                else:
                    left_state='open'
                
                # 左眼状态画图    
                left_eye_x = int(points.T[max_face_id][0])
                left_eye_y = int(points.T[max_face_id][5])
                cv2.putText(draw, "{}".format(left_state), (left_eye_x-30, left_eye_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1, 3)
                
                # 判断右眼状态
                if get_eye_label(eye_right_region) == 'closed_eye':
                    right_state='closed'
                else:
                    right_state='open'
                
                # 右眼状态画图
                right_eye_x = int(points.T[max_face_id][1])
                right_eye_y = int(points.T[max_face_id][6])
                cv2.putText(draw, "{}".format(right_state), (right_eye_x-30, right_eye_y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, 3)
                
                
                # 判断嘴部状态
                mouth_label = get_mouth_label(mouth_region)
                if mouth_label == 'closed_mouth':
                    mouth_state='closed'
                    mouth_states.append(0)
                else :
                    mouth_state='open'
                    mouth_states.append(1)
                
                # 嘴部状态画图
                mouth_x = int((points.T[max_face_id][4]-points.T[max_face_id][3])/2+points.T[max_face_id][3])
                mouth_y = int((points.T[max_face_id][9] + points.T[max_face_id][8])/2)
                cv2.putText(draw, "{}".format(mouth_state), (mouth_x-30, mouth_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 3)
                cv2.imshow("img", draw)

                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 如果左眼和右眼都闭合，增加闭合帧计数
                if left_state == 'closed' and right_state == 'closed':
                    closed_frames += 1

                # 如果超过时间窗口，计算 PERCLOS 并重置计数器
                if total_frames >= frame_window:
                    perclos = (closed_frames / total_frames) * 100
                    print("PERCLOS: {:.2f}%".format(perclos))
                    
                    if perclos > 35:
                        fatigue = 1

                    # 重置计数器
                    total_frames = 0
                    closed_frames = 0
            
            end = time.time()
            times.append(end-start)
            #print("it takes ", end-start, " to solve one frame")
        
        #print(times)
        average = np.mean(times[1:])
        print("Average time:", average)

        # 绘制处理速率图像
        # plt.plot(times[1:])
        # plt.xlabel('frame')
        # plt.ylabel('s/frame')
        # plt.hlines(average, 0, len(times)-2, colors='r', linestyles='dashed')  # 画出平均值的线
        # plt.show()
        
        count = 1
        max_cnt = 0
        #print(mouth_states)
        for element in mouth_states:
            if element == 1 :
                count += 1
            else:
                max_cnt = max(max_cnt, count)
                count = 1
        max_cnt = max(max_cnt, count)
        
        print("consistent open mouth frames ", max_cnt)
        
        if max_cnt > 40:
            fatigue = 1

        if fatigue:
            cv2.putText(draw, "fatigue", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2, 3)
            print("fatigue detected, rest now")
        else:
            cv2.putText(draw, "awake", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2, 3)
            print("awake detected")
        cv2.imshow("img", draw)
        cv2.waitKey(0)