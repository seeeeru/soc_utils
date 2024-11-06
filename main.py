from utils import read_video, save_video
from trackers import Tracker
import pickle
import cv2
import numpy as np
import pandas as pd
from team_assigner import TeamAssigner
from minimap import Minimap
from draw_line import Drawline
from player_ball_assigner import PlayerBallAssigner

import os

def main():
    video_frames = read_video('input_videos/white_yellow_input.mp4')

    minimap = Minimap("models/best_key_point.pt")
    tracker = Tracker('models/yolov8x_player.pt', 'models/ball_pt_0602.pt')
    player_assigner = PlayerBallAssigner()
    draw_line = Drawline()
    team_assigner = TeamAssigner()

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/_track_stubs_deep.pkl')
    tracks = tracker.plus_ball(video_frames,tracks,read_from_stub=True,stub_path= 'stubs/_track_stubs_ball.pkl')
    keypoints = minimap.get_object_keypoints(video_frames, read_from_stub=True, stub_path='stubs/_keypoints_stubs.pkl')

    # 트랙에 위치 정보 추가
    tracker.add_position_to_tracks(tracks)
    h = minimap.get_h(video_frames, keypoints)
    minimap.add_transfromed_position(tracks, h)
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # 모든 프레임에 대해 팀 할당 및 색상 추가
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    no_draw_frames = tracker.draw_annotations(video_frames, tracks)
    minimap_frames = minimap.draw_minimap(video_frames,tracks)
    no_draw = minimap.combine_frames(no_draw_frames,minimap_frames)

    save_video(no_draw, 'output_videos/no_draw.mp4')
    os.system(f'ffmpeg -i output_videos/no_draw.mp4 -vcodec libx264 output_videos/no_draw_change.mp4')
    
    
    target_ids = [4,22,9,30]
    specific_postions = draw_line.get_positions_for_specific_track_ids(tracks, target_ids)
    transformed_positions = draw_line.transform_positions(specific_postions,tracks)
    
    draw_frames = draw_line.draw_lines_between_tracks(no_draw_frames, specific_postions,transformed_positions, target_ids)
    draw = minimap.combine_frames(draw_frames,minimap_frames)
    

    
    save_video(draw,'output_videos/draw.mp4')
    os.system(f'ffmpeg -i output_videos/draw.mp4 -vcodec libx264 output_videos/draw_change.mp4')

    df_possesion = player_assigner.get_df_ball(tracks)
    df_jumyu = df_possesion[['frame_number','ratio_1','ratio_2']]
    new_df = []
    for i in range(0, df_jumyu.shape[0], 24):
        df_slice = df_jumyu.iloc[i:i+24]
        # 첫 번째 행만 선택하여 새로운 데이터프레임에 추가
        new_df.append(df_slice.iloc[0])

    new_df = pd.DataFrame(new_df)
    new_df['frame_number'] = new_df['frame_number'] / 24
    new_df = new_df.iloc[1:]


    tracker.add_velo_dis(tracks)
    df_velo_dis = tracker.get_df(tracks)
    dfs = [df_velo_dis,df_possesion,new_df]
    with open('stubs/df.pkl','wb') as f:
        pickle.dump(dfs, f)
    
   
if __name__ == '__main__':
    main()