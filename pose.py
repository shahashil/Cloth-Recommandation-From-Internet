import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


    
def filter_images(filepath,image_no):
    # For static images:
    filepath = 'E:\Charuset Pykathin\Code\downloads'#+filepath
    IMAGE_FILES = os.listdir(filepath)

    #IMAGE_FILES = ["E:\Charuset Pykathin\Code\downloads\men pants with blue t shirt graphic\pan.jpg"]
    BG_COLOR = (192, 192, 192) # gray

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        try:
            image = cv2.imread(filepath+'\\'+file)    
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #recolor back to BRG
            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            landmark = results.pose_landmarks.landmark
            nose           = [landmark[mp_pose.PoseLandmark.NOSE.value].x,landmark[mp_pose.PoseLandmark.NOSE.value].y,landmark[mp_pose.PoseLandmark.NOSE.value].visibility]
            left_shoulder  = [landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility]
            right_shoulder = [landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility]
            left_ankle     = [landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility]
            right_ankle    = [landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility]

            marks = [left_shoulder,right_shoulder,left_ankle,right_ankle,nose]
            if left_shoulder[2] > 0.6 and right_shoulder[2] > 0.6 and left_ankle[2] > 0.9 and right_ankle[2] > 0.9 and nose[2] > 0.6:
                image_no+=1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("Dataset/"+str(image_no)+'.jpg',image)
                print("done")
            else:
                pass
        except:
            pass    
    return image_no

if __name__ == '__main__':
    get_pose("1.jpg",0)