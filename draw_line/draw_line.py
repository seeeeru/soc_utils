import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance ,get_foot_position
from collections import defaultdict, deque

class Drawline():
    
    def get_positions_for_specific_track_ids(self, tracks, target_ids):
        specific_position = {}
        for object_name, object_tracks in tracks.items():
            if object_name != "players":
                continue
            for frame_num,frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    if track_id in target_ids and 'position' in track_info:
                        if frame_num not in specific_position:
                            specific_position[frame_num] = {}
                        if track_id not in specific_position[frame_num]:
                            specific_position[frame_num][track_id] = track_info['position']
        return specific_position
    
    def draw_lines_between_tracks(self, frames, specific_positions, transformed_positions, target_ids):
        output_frames = []
        speed_records = defaultdict(lambda: deque(maxlen=12))
        for frame_num, frame_positions in specific_positions.items():
        # 현재 프레임 번호에 해당하는 프레임을 가져옴
            frame = frames[frame_num]

            for id in target_ids:
                # 해당 객체가 현재 프레임의 transformed_positions에 없다면 건너뜀
                if id not in transformed_positions[frame_num]:
                    continue
            
            for i in range(len(target_ids) - 1):
                start_id = target_ids[i]
                end_id = target_ids[i+1]
            
                if start_id not in frame_positions or end_id not in frame_positions:
                    continue
            
                start_point = frame_positions[start_id]
                end_point = frame_positions[end_id]
                
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
            
                if transformed_positions and frame_num in transformed_positions:
                    if start_id in transformed_positions[frame_num]:
                        start_point_2 = transformed_positions[frame_num][start_id]['transformed_position']
                    if end_id in transformed_positions[frame_num]:
                        end_point_2 = transformed_positions[frame_num][end_id]['transformed_position']
                        
                distance = np.sqrt((end_point_2[0] - start_point_2[0])**2 + (end_point_2[1] - start_point_2[1])**2)
                
                mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv2.putText(frame, f"{distance:.2f} m", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)
        
        return output_frames
    
    def transform_positions(self, specific_positions,tracks):
        transformed_positions = {}
        previous_positions = {}
        total_distances = {}
        velo = {}

        for frame_num, frame_data in specific_positions.items():
            transformed_positions[frame_num] = {} 
            for track_id, position in frame_data.items():
                transformed_position = tracks['players'][frame_num][track_id]['for_distance']
                if transformed_position is not None:
                    current_position = transformed_position
                    if track_id in previous_positions:
                        prev_position = previous_positions[track_id]
                        move_dis = ((current_position[0] - prev_position[0]) ** 2 +
                                    (current_position[1] - prev_position[1]) ** 2) ** 0.5
                        velo = move_dis*36
                    else: 
                        move_dis = 0.0
                        velo = 0.0
                    previous_positions[track_id] = current_position

                    if track_id in total_distances:
                        total_distances[track_id] += move_dis
                    else:
                        total_distances[track_id] = move_dis

                    transformed_positions[frame_num][track_id] = {
                        'transformed_position': transformed_position,
                        'distance': move_dis,
                        'total_distance': total_distances[track_id]
                        }
                    transformed_positions[frame_num][track_id]['velocity'] = velo

                    


        return transformed_positions