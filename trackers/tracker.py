from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from deep_sort_realtime.deepsort_tracker import DeepSort



class Tracker:
    def __init__(self, model_path, model_path_b):
        self.model = YOLO(model_path)
        self.model_b = YOLO(model_path_b)
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        self.tracker2 = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        detections_b = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.7)
            detections += detections_batch
            detections_batch_ball = self.model_b.predict(frames[i:i+batch_size], conf=0.1)
            detections_b += detections_batch_ball
        return detections, detections_b

 

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections,detections_b = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]


            detection_with_tracks = self.tracker2.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
       
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def plus_ball(self, frames,tracks,read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        detections,detections_b = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections_b):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            tracks["ball"].append({})

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        
        return tracks

    
    def number_box(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame
        

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.number_box(frame, player["bbox"], color, track_id)

            for _, referee in referee_dict.items():
                frame = self.number_box(frame, referee["bbox"], (0, 255, 255))


            for _, ball in ball_dict.items():
                frame = cv2.circle(frame, ball['position'],radius= 5,color= (0, 0, 255),thickness=-1)

            output_video_frames.append(frame)

        return output_video_frames
    
    def add_velo_dis(self, tracks):
        previous_positions = {}
        total_distances = {}
        velo = {}
        for frame_num, player_track in enumerate(tracks['players']):
            for track_id, track_info in player_track.items():
                coord = tracks['players'][frame_num][track_id]['for_distance']
                if coord is not None:
                    curr_pos = coord
                    if track_id in previous_positions:
                        prev_position = previous_positions[track_id]
                        move_dis = ((curr_pos[0] - prev_position[0]) ** 2 +
                                    (curr_pos[1] - prev_position[1]) ** 2) ** 0.5
                        velo = move_dis*36
                    else: 
                        move_dis = 0.0
                        velo = 0.0
                    previous_positions[track_id] = curr_pos

                    if track_id in total_distances:
                        total_distances[track_id] += move_dis
                    else:
                        total_distances[track_id] = move_dis

                    tracks['players'][frame_num][track_id] = {
                        'transformed_position': curr_pos,
                        'distance': move_dis,
                        'total_distance': total_distances[track_id]
                        }
                    tracks['players'][frame_num][track_id]['velocity'] = velo

        return tracks
    
    def get_df(self, tracks, frame_interval=8):
        # 각 track_id 별로 데이터를 저장할 딕셔너리 초기화
        track_data = {}

        # 각 프레임에 대해 순회
        for frame_num, player_tracks in enumerate(tracks['players']):
            for track_id, track_info in player_tracks.items():
                # track_id가 track_data에 없으면 초기화
                if track_id not in track_data:
                    track_data[track_id] = {'frame': [], 'total_distance': [], 'velocity': []}
                
                # 현재 프레임, total_distance, velocity 추가
                track_data[track_id]['frame'].append(frame_num)
                track_data[track_id]['total_distance'].append(track_info.get('total_distance', 0.0))
                track_data[track_id]['velocity'].append(track_info.get('velocity', 0.0))
        
        # 각 track_id 별로 12프레임 단위로 누적 및 평균 데이터프레임 생성
        aggregated_dataframes = {}
        for track_id, data in track_data.items():
            df = pd.DataFrame(data)
            df['seconds'] = df['frame'] / 24  # 프레임을 초 단위로 변환

            aggregated_data = {
                'seconds': np.arange(0, len(df) / 24, frame_interval / 24),
                'total_distance': [],
                'velocity': []
            }

            for i in range(0, len(df), frame_interval):
                end_idx = min(i + frame_interval, len(df))
                frame_slice = df.iloc[i:end_idx]
                aggregated_data['total_distance'].append(frame_slice['total_distance'].iloc[-1])
                aggregated_data['velocity'].append(frame_slice['velocity'].mean())
            
            aggregated_dataframes[track_id] = pd.DataFrame(aggregated_data)
        
        return aggregated_dataframes

