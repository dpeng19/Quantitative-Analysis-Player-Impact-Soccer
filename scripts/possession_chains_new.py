# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:28:20 2025

@author: Derek
"""

import soccerdata as sd
import pandas as pd
import json
# plotting
import os
import pathlib
import warnings 
import xgboost
from joblib import load
from mplsoccer import Pitch
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from pathlib import Path
import socceraction
from socceraction.atomic import spadl
from socceraction.spadl.schema import SPADLSchema
import socceraction.spadl.opta
import pandas as pd

import sys
import os
import importlib

# Build path to 'helpers' directory
cwd = os.getcwd()
helpers_path = os.path.join(cwd, 'helpers')

# Add to sys.path so Python can find your modules
sys.path.append(helpers_path)

# Import helpers
import possession_chains
import scrape_event_data
import utils

#reload when necessary
importlib.reload(scrape_event_data)
importlib.reload(possession_chains)
importlib.reload(utils)




#24/25 epl event data
season_events = scrape_event_data.get_event_data(
    leagues='ENG-Premier League', 
    seasons=2024, 
)
events_spadl = scrape_event_data.get_event_data(
    leagues='ENG-Premier League', 
    seasons=2024, 
    output_fmt='spadl',
)

#more info into the actions
merged_df = pd.merge(
    events_spadl,
    season_events[['game_id', 'event_id', 'minute', 'second', 'expanded_minute', 'type', 'qualifiers']],
    left_on=['game_id', 'original_event_id'],
    right_on=['game_id', 'event_id'],
    how='left'
)

# Define pass-like actions (first 7 types)
pass_like_ids = [0, 1, 2, 3, 4, 5, 6]

# Define shot types
shot_ids = [11, 12, 13]

non_offensive_ids = [9, 10, 18, 21]
# Define goalkeeper actions
gk_ids = [14, 15, 16, 17]
non_offensive_actions = merged_df[merged_df.type_id.isin(non_offensive_ids)]
gk_actions = merged_df[merged_df.type_id.isin(gk_ids)]
#check
non_offensive_actions.type.value_counts()
gk_actions.type.value_counts()
#drop non-offensive and goalkeeper actoins
df = merged_df.drop(non_offensive_actions.index)
df = df.drop(gk_actions.index)
df.type.value_counts()
df["prev_type_id"] = df.shift(1, fill_value=0)["type_id"]
df["prev_result_id"] = df.shift(1, fill_value=0)["result_id"]
df["prev_team_id"] = df.shift(1, fill_value=0)["team_id"]
df["next_type_id"] = df.shift(-1, fill_value=0)["type_id"]
df["next_result_id"] = df.shift(-1, fill_value=0)["result_id"]
df["next_team_id"] = df.shift(-1, fill_value=0)["team_id"]
df["next_period_id"] = df.shift(-1, fill_value=0)["period_id"]


#shots = events_spadl[events_spadl.type_id == 11]
#shots_reg = season_events[season_events.type == 'Shot']
#shots_reg = season_events[(season_events['type'] == 'Goal') | (season_events['type'].str.contains('Shot')) ]

#df = df.drop(wierd_21.index)
out_plays = df.loc[(df['end_x'] == 0) | (df['end_x'] == 105) | (df['end_y'] == 0) | (df['end_y'] == 68)]

#df.type_id.value_counts()
#non-shot kick-outs
df['kicked_out'] = (((df['end_x'] == 0) | (df['end_x'] == 105) | (df['end_y'] == 0) | (df['end_y'] == 68)) 
                    & (~df['type_id'].isin(shot_ids)))
df['kicked_out'].value_counts()

#isolate possession chains
poss = possession_chains.isolate_chains(df)

#---- reorder columns ----
col1 = 'possesion_chain'
col2 = 'possesion_chain_team'
col3 = 'type_id'
col4 = 'type'
cols = poss.columns.tolist()

cols.remove(col1)
cols.remove(col2)
cols.remove(col3)
cols.remove(col4)

cols.insert(6, col1)
cols.insert(7, col2)
cols.insert(8, col3)
cols.insert(9, col4)

# Reorder the DataFrame
poss2 = poss[cols]
poss2['team_equal'] = poss2['possesion_chain_team'] == poss2['team_id']
#check to make sure everything is right
poss2['team_equal'].value_counts()


shots_merge = merged_df[merged_df.type_id.isin([11, 12, 13])]
shots_merge = shots_merge.reset_index(drop=True)

#---- ensure the types of shots look right, throw error otherwise ----
# Define the allowed shot types
allowed_types = {'SavedShot', 'MissedShots', 'Goal', 'ShotOnPost'}

# Get the actual types present in the DataFrame
actual_types = set(shots_merge['type'].unique())

# Check for unexpected types
unexpected_types = actual_types - allowed_types

if unexpected_types:
    raise ValueError(f"Unexpected shot types found: {unexpected_types}")


#---- pull shots from understat, which has expected goals, want to match the shots ----
us = sd.Understat("ENG-Premier League", seasons=2024)
season_shots = us.read_shot_events()
sorted_shots = season_shots.groupby('game_id', sort = False, group_keys=False).apply(lambda x: x.sort_values('minute')).reset_index(drop=True)
#remove own goals
sorted_shots = sorted_shots.loc[sorted_shots['result'] != 'Own Goal']
player_season_stats = us.read_player_season_stats()
player_season_stats = player_season_stats.reset_index()
#merge to add player name info to understat shots
merged_shots = pd.merge(
    sorted_shots,
    player_season_stats[['player_id', 'player']],
    on = ['player_id'],
    how='left'
)

#standardize player names across two sources for shots
player_mapping = utils.get_player_mappings(shots_merge, merged_shots)


#pull schedules to standardize game_id
understat_schedule = us.read_schedule().reset_index()

ws = sd.WhoScored("ENG-Premier League", seasons=2024)
whoscored_schedule = ws.read_schedule().reset_index()

#get only date
understat_schedule['date_only'] =  understat_schedule.game.apply(lambda x: x.split(' ')[0])
whoscored_schedule['date_only'] = whoscored_schedule.game.apply(lambda x: x.split(' ')[0])



# ---- check for valid dates in the schedules----
for df in [understat_schedule, whoscored_schedule]:
    if not df['date_only'].apply(utils.is_valid_date).all():
        raise ValueError("One or more dates are invalid.")

print("All dates are valid.")


#---- standardize game_id column to prepare for merging two dataframes ----
#set game_idx to index for whoscored, easy to standardize
whoscored_schedule['game_idx'] = whoscored_schedule.index
whoscored_schedule, understat_schedule = utils.get_standardized_game_idx(
    whoscored_schedule, 
    understat_schedule
)
merged_shots = pd.merge(
    merged_shots,
    understat_schedule[['game_id', 'game_idx', 'home_team', 'away_team']],
    on = 'game_id',
    how = 'left'
)

shots_merge = pd.merge(
    shots_merge,
    whoscored_schedule[['game_id', 'game_idx', 'home_team', 'away_team']],
    on = 'game_id', 
    how = 'left'
)       

#---- check if game_id, player, and minute uniquely identifies a shot event ----
merge_shots_group = merged_shots.groupby(["game_idx", "player", "minute"]).size().reset_index(name = 'count')

#same = merge_shots_group.loc[merge_shots_group['count'] > 1]
#shots that are not uniquely identified by game_id, player, and minute
'''
check_same_shot= pd.merge(
    merged_shots,
    same,
    on = ['game_idx', 'player', 'minute']
    
    )
'''
#scale to whoscored coordinate system
merged_shots['scale_x'] = merged_shots['location_x'] * 105
merged_shots['scale_y'] = merged_shots['location_y'] * 68

shots_merge['xg'] = -1.0
shots_merge['minute_2'] = -1
shots_merge['player_2'] = ''
shots_merge['x_2'] = -1.0
shots_merge['y_2'] = -1.0
for i in range(len(player_mapping)):
    #use mapping previously created to get only players with this name
    shots_1 = shots_merge.loc[shots_merge.player == player_mapping.loc[i, "player"]].copy().sort_values(by = ['game_idx', 'minute'])
    shots_2 = merged_shots.loc[merged_shots.player == player_mapping.loc[i, "name_2"]].copy().sort_values(by = ['game_idx', 'minute'])
    shots_2['check'] = 0
    for j, row in shots_1.iterrows():
        check = 0
        for k, row2 in shots_2.iterrows():
            if row2["check"] != 0:
                continue
            if row2["game_idx"] > row["game_idx"]:
                break
            if row2["game_idx"] == row["game_idx"] and row2["minute"] == row["minute"]:
                if merge_shots_group.loc[(merge_shots_group.game_idx == row2["game_idx"]) & 
                    (merge_shots_group.player == player_mapping.loc[i, "name_2"]) & (merge_shots_group.minute == row2["minute"])]['count'].values[0] > 1:
                    #candidate shots 
                    poss_shots = shots_2.loc[(shots_2.game_idx == row2["game_idx"]) & 
                         (shots_2.minute == row2["minute"])][['scale_x', 'scale_y']]
                    poss_shots['dist_away'] = (poss_shots['scale_x'] - row["start_x"])**2 + (poss_shots['scale_y'] - row["start_y"])**2
                    closest_shot = poss_shots.loc[poss_shots['dist_away'].idxmin()]
                    closest_idx = poss_shots['dist_away'].idxmin()
                    
                    check = 1
                    shots_2.at[closest_idx, "check"] = 1
                    shots_merge.at[j, "xg"] = shots_2.loc[closest_idx, "xg"]
                    shots_merge.at[j, "minute_2"] = shots_2.loc[closest_idx, "minute"]
                    shots_merge.at[j, "player_2"] = shots_2.loc[closest_idx, "player"]
                    shots_merge.at[j, "x_2"] = shots_2.loc[closest_idx, "scale_x"]
                    shots_merge.at[j, "y_2"] = shots_2.loc[closest_idx, "scale_y"]
                    break
                else:
                    check = 1
                    shots_2.at[k, "check"] = 1
                    #get xg
                    shots_merge.at[j, "xg"] = row2["xg"]
                    #get additional info
                    shots_merge.at[j, "minute_2"] = row2["minute"]
                    shots_merge.at[j, "player_2"] = row2["player"]
                    shots_merge.at[j, "x_2"] = row2["scale_x"]
                    shots_merge.at[j, "y_2"] = row2["scale_y"]
                    break
        if check == 0:
            for k, row2 in shots_2.iterrows():
                if row2["check"] != 0:
                    continue
                if row2["game_idx"] > row["game_idx"]:
                    break
                if row2["game_idx"] == row["game_idx"] and ((row2["minute"] == row["minute"] + 1 and row['second'] >= 55) or (row2["minute"] == row["minute"] - 1 and row['second'] <= 5)):
                    
                    if merge_shots_group.loc[(merge_shots_group.game_idx == row2["game_idx"]) & 
                        (merge_shots_group.player == player_mapping.loc[i, "name_2"]) & (merge_shots_group.minute == row2["minute"])]['count'].values[0] > 1:
                        #candidate shots 
                        poss_shots = shots_2.loc[(shots_2.game_idx == row2["game_idx"]) & 
                             (shots_2.minute == row2["minute"])][['scale_x', 'scale_y']]
                        poss_shots['dist_away'] = (poss_shots['scale_x'] - row["start_x"])**2 + (poss_shots['scale_y'] - row["start_y"])**2
                        closest_shot = poss_shots.loc[poss_shots['dist_away'].idxmin()]
                        closest_idx = poss_shots['dist_away'].idxmin()
                        check = 1
                        shots_2.at[closest_idx, "check"] = 1
                        shots_merge.at[j, "xg"] = shots_2.loc[closest_idx, "xg"]
                        shots_merge.at[j, "minute_2"] = shots_2.loc[closest_idx, "minute"]
                        shots_merge.at[j, "player_2"] = shots_2.loc[closest_idx, "player"]
                        shots_merge.at[j, "x_2"] = shots_2.loc[closest_idx, "scale_x"]
                        shots_merge.at[j, "y_2"] = shots_2.loc[closest_idx, "scale_y"]
                        break
                            
                            

                    else:
                        check = 1
                        shots_2.at[k, "check"] = 1
                        #get xg
                        shots_merge.at[j, "xg"] = row2["xg"]
                        #get additional info
                        shots_merge.at[j, "minute_2"] = row2["minute"]
                        shots_merge.at[j, "player_2"] = row2["player"]
                        shots_merge.at[j, "x_2"] = row2["scale_x"]
                        shots_merge.at[j, "y_2"] = row2["scale_y"]
                        
 
#how many non-matches
print(len(shots_merge[shots_merge.xg < 0])) 
#shots that have no xg, will need to calculate ourselves
no_match = shots_merge[shots_merge.xg < 0] 


#---- debug -----
'''
not_equal = poss2[poss2.team_equal == 0]
season_events_1821101 = season_events[season_events.game_id == 1821101]
'''
own_goals = poss2[poss2.result_id == 3]
shots = poss2[poss2.type_id.isin(shot_ids)]
shots_normal = season_events[season_events.type == '']
#head is 1, seems like 5 is foot, 4 is other, 2 is none
shots.bodypart_id.value_counts()
#df['kicked_out'].value_counts()
#out_plays = df.loc[(df['end_x'] == 0) | (df['end_x'] == 105) | (df['end_y'] == 0) | (df['end_y'] == 68)]
def calulatexG(df):
    """
    Parameters
    ----------
    df : dataframe
        dataframe with Wyscout event data.

    Returns
    -------
    xG_sum: dataframe
        dataframe with xG for each shot

    """
    #very basic xG model based on
    shots = df.loc[df["type_id"].isin(shot_ids)].copy()
    shots["X"] = 105 - shots['start_x']
    shots["Y"] = shots['start_y']
    shots["C"] = abs(shots['start_y']- 34)
    #calculate distance and angle
    shots["Distance"] = np.sqrt(shots["X"]**2 + shots["C"]**2)
    shots["Angle"] = np.where(np.arctan(7.32 * shots["X"] / (shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * shots["X"] /(shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)), np.arctan(7.32 * shots["X"] /(shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) + np.pi)
   #if you ever encounter problems (like you have seen that model treats 0 as 1 and 1 as 0) while modelling - change the dependant variable to object
    shots["Goal"] = shots['result_id'].astype(object)
        #headers have id = 403
    headers = shots.loc[shots.bodypart_id == 1]
    non_headers = shots.drop(headers.index)

    headers_model = smf.glm(formula="Goal ~ Distance + Angle" , data=headers,
                               family=sm.families.Binomial()).fit()
    #non-headers
    nonheaders_model = smf.glm(formula="Goal ~ Distance + Angle" , data=non_headers,
                               family=sm.families.Binomial()).fit()
    #assigning xG
    df["xG"] = 0
    b_head = headers_model.params
    xG = 1/(1+np.exp(b_head[0]+b_head[1]*headers['Distance'] + b_head[2]*headers['Angle']))
    headers = headers.assign(xG = xG)
    for index, row in headers.iterrows():
        df.at[index, "xG"] = row["xG"]
    #non-headers
    b_nhead = nonheaders_model.params
    xG = 1/(1+np.exp(b_nhead[0]+b_nhead[1]*non_headers['Distance'] + b_nhead[2]*non_headers['Angle']))
    non_headers = non_headers.assign(xG = xG)
    for index, row in non_headers.iterrows():
        df.at[index, "xG"] = row["xG"]

    penalties = df.loc[df.type_id == 12]
    #treating penalties like shots
    penalties["X"] = 105 - penalties['start_x']
    #calculate distance and angle
    penalties["Distance"] = 105 - penalties['start_x']
    penalties["Angle"] = np.arctan(7.32 * penalties["X"] /(penalties["X"]**2 - (7.32/2)**2))
    #if you ever encounter problems (like you have seen that model treats 0 as 1 and 1 as 0) while modelling - change the dependant variable to object
    penalties["Goal"] = penalties['result_id'].astype(object)
    
    penalties_model = smf.glm(formula="Goal ~ Distance + Angle" , data=penalties, 
                               family=sm.families.Binomial()).fit()
    b_penalty = penalties_model.params
    xG = 1/(1+np.exp(b_penalty[0]+b_penalty[1]*penalties['Distance'] + b_penalty[2]*penalties['Angle'])) 
    
    penalties = penalties.assign(xG = xG)
    for index, row in penalties.iterrows():
        df.at[index, "xG"] = row["xG"]
    return df

df = calulatexG(poss2)
#investigate a chain
df.loc[df["possesion_chain"].isin([3,4])][["eventName", "possesion_chain", "xG"]]
#check if shots are unique per minute in match

