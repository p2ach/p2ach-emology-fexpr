# This is a sample Python script.
import os
import cv2
import csv
from utils import FaceRun
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

DATASET_FOLDER = "/home/doy/dataset/fexpr/"
DST_PATH = "/home/doy/dataset/fexpr/dest/listup/"

def filter_headers(lines):
    if lines[8].split(' ')[0] == "Video":
        return lines[8:]
    else:
        raise ("not match pattern")

def convert_text2df(path2client,faceEngine):
    path2parent=os.path.join(DATASET_FOLDER,path2client)
    childs=os.listdir(path2parent)
    idx=0
    for child in childs:
        if child.endswith('.txt'):
            with open(os.path.join(path2parent,child),'r') as file:
                lines = file.readlines()
            valid_lines_labels = filter_headers(lines)
            video_name=child.split('.')[0].split('_')[-1]+'.webm'
            cap = cv2.VideoCapture(os.path.join(path2parent,video_name))
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            for _line in valid_lines_labels:
                label=_line.split('\n')[0].split('\t')
                path2csv = os.path.join(DST_PATH,'labels','{:016d}.csv'.format(idx))
                with open(path2csv, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(label)
                ret, frame = cap.read()
                face_img = faceEngine.get_detection(frame,frame_width,frame_height)
                if face_img is not None:
                    face_landmark=faceEngine.get_landmark(face_img)
                    try:
                        if face_landmark is not None:
                            path2np = os.path.join(DST_PATH, 'images', '{:016d}.np'.format(idx))
                            np.save(path2np,face_landmark)
                            print("idx :",idx)
                            idx += 1
                        else:
                            pass
                    except Exception as e:
                        print(e)
                else:
                    pass
















def run_train(dataset_folder):
    tests=os.listdir(dataset_folder)
    faceEngine = FaceRun()
    for test in tests:
        convert_text2df(test,faceEngine)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    run_train(DATASET_FOLDER)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
