import numpy as np
import pandas as pd

import cv2
import skimage
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import json
import yaml
import time
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2

class Minimap():
    def __init__(self,model_path):
        self.minimap = cv2.imread('./minimap/MINIMAP.png')
        self.model_k = YOLO(model_path)
        #여기서 어느정도의 변화가 있을 때 호모그래피를 업데이트 할 것인지 결정합니다.
        self.keypoints_displacement_mean_tol = 5
        #키포인트 제이슨
        with open('./minimap/keypoint.json', 'r') as f:
            self.keypoints_map_pos = json.load(f)
        #키포인트 야믈파일
        with open('./minimap/keyloint.yaml', 'r') as file:
            classes_names_dic = yaml.safe_load(file)
        self.classes_names_dic = classes_names_dic['names']
        #변환행렬 h를 구하고 프레임별로 리스트에 저장하는 함수입니다.
        self.minimap_w = 451
        self.minimap_h = 257
        self.real_w = 105
        self.real_h = 68
        self.ratio_w = self.real_w/self.minimap_w
        self.ratio_h = self.real_h/self.minimap_h
    
    def get_object_keypoints(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                keypoints = pickle.load(f)
            return keypoints
        
        keypoints = []

        for frame in frames:
            keypoints = self.model_k(frame, conf=0.1)
            frame_detections = []
            for det in keypoints:
                if det.keypoints is not None:
                    frame_detections.append(det.detections.tolist())
            keypoints.append(frame_detections)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(keypoints, f)

            return keypoints


    def add_transfromed_position(self, tracks,h):
        for frame_num, player_track in enumerate(tracks['players']):
            h_ = h[frame_num]            
            for track_id, track_info in player_track.items():
                before_t = track_info['position']
                #호모그래피를 계산하기 위해서는 3행이 필요합니다(행렬곱). 그래서 1로 이루어진 3열을 추가 한 후에 그 행렬을 transpose해서 계산해요.
                before_t = np.append(np.array(before_t), np.array([1]), axis=0)                   
                after_t = np.matmul(h_, np.transpose(before_t))                              
                after_t = after_t/after_t[2]
                #array[319,324]이런식으로 저장이 되기 때문에 깔끔하게 정리하기 위해서 tuple함수를 사용합니다. (319,324)형태로 정리돼요. 따로 round처리는 하지 않았습니다.
                after_t = np.transpose(after_t)[:2]
                for_distance = [after_t[0]*self.ratio_w,after_t[1]*self.ratio_h]
                tracks['players'][frame_num][track_id]['for_distance'] = tuple(for_distance)
                after_t = tuple(np.round(after_t).astype(int))
                tracks['players'][frame_num][track_id]['after_trans'] = after_t

                #이 함수는 main.py에서 바로 tracks에 밸류값을 추가할 거라 return값이 필요하지 않아요.

    def draw_minimap(self, frames, tracks):
        minimap_frames = []
        for frame_num,frame in enumerate(frames):
            minimap_c = self.minimap.copy()

            minimap_dict = tracks['players'][frame_num]
            for track_id,player in minimap_dict.items():
                colors = player.get("team_color",(0,0,255))
                minimap_c = cv2.circle(minimap_c, player['after_trans'],radius= 5,color= colors,thickness=-1)
            minimap_frames.append(minimap_c)
                
        return minimap_frames
    
    
    def combine_frames(self,output_frames, minimap_frames):
        combined_frames = []
        for output_frame, minimap_frame in zip(output_frames, minimap_frames):
            oh, ow, _ = output_frame.shape
            mh, mw, _ = minimap_frame.shape

            scale = 0.25
            new_mw = int(ow * scale)
            new_mh = int(mh * new_mw / mw)
            minimap_frame_resized = cv2.resize(minimap_frame, (new_mw, new_mh))

            x_offset = ow - new_mw
            y_offset = oh - new_mh

            combined_frame = output_frame.copy()
            combined_frame[y_offset:y_offset + new_mh, x_offset:x_offset + new_mw] = minimap_frame_resized

            combined_frames.append(combined_frame)

        return combined_frames
    

    def get_h(self,frames,keypoints):
        #mat에는 프레임별로 계산된 변환행렬h가 추가되게 만들었어요. 이게 마지막에 반환되는 리스트입니다.
        mat = []
        for i in range(0,len(frames)):
            #keypoints = self.get_object_keypoints(frames, read_from_stub=True, stub_path='/Users/chan/tennis/football/stubs/_keypoints_stubs.pkl')                    
            bboxes_k_c = keypoints[0].boxes.xywh.cpu().numpy()                       
            labels_k = list(keypoints[0].boxes.cls.cpu().numpy())   
            detected_labels = [self.classes_names_dic[i] for i in labels_k]
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])
            detected_labels_dst_pts = np.array([self.keypoints_map_pos[i] for i in detected_labels])


            if len(detected_labels) > 3 and len(set(detected_labels)) == len(detected_labels):
                if i > 0:                    
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]  
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  
                        update_homography = coor_error > self.keypoints_displacement_mean_tol                 
                    else:
                        update_homography = True
                else:
                    update_homography = True

                if  update_homography:
                    h, mask = cv2.findHomography(detected_labels_src_pts,                 
                                                detected_labels_dst_pts)
                #변환행렬을 도출 한 후에, 다음 프레임에서 쓰일 좌표들을 저장합니다.
                detected_labels_prev = detected_labels.copy()                             
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()
            mat.append(h)
        return mat

        
    