import sys 
import pandas as pd
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player
    
    def get_df_ball(self,tracks):
        team_ball_control=[]
        for frame_num, player_track in enumerate(tracks['players']):
            if 1 in tracks["ball"][frame_num]:
                ball_bbox = tracks["ball"][frame_num][1]["bbox"]
                assigned_player = self.assign_ball_to_player(player_track, ball_bbox)


                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])


                else:
                    if team_ball_control:
                        team_ball_control.append(team_ball_control[-1])

                    else:
                        team_ball_control.append('unknown')

            else:
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])

                else:
                    team_ball_control.append('unknown')

        team_ball_control = np.array(team_ball_control)
        team_ball_c=pd.DataFrame({
        'frame_number': range(len(team_ball_control)),
        'team_ball_control': team_ball_control})
        team_ball_c['team_ball_control'] = pd.to_numeric(team_ball_c['team_ball_control'], errors='coerce')
        team_ball_c['cumsum_1'] = (team_ball_c['team_ball_control'] == 1).cumsum()
        team_ball_c['cumsum_2'] = (team_ball_c['team_ball_control'] == 2).cumsum()
        team_ball_c['ratio_1'] = round(team_ball_c['cumsum_1'] / (team_ball_c['cumsum_1'] + team_ball_c['cumsum_2']),3)
        team_ball_c['ratio_2'] = round(team_ball_c['cumsum_2'] / (team_ball_c['cumsum_1'] + team_ball_c['cumsum_2']),3)
        team_ball_c['ratio_1'].fillna(0, inplace=True)
        team_ball_c['ratio_2'].fillna(0, inplace=True)
        final_team_ball_c = team_ball_c[['frame_number', 'team_ball_control', 'ratio_1', 'ratio_2']]
        df_jumyu = final_team_ball_c[['frame_number','ratio_1','ratio_2']]

        return final_team_ball_c