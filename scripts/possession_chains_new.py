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

#reload when necessary
importlib.reload(scrape_event_data)
importlib.reload(possession_chains)




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

out_plays['type_id'].value_counts()
out_shots = out_plays[out_plays.type_id.isin(shot_ids)]
out_actions = out_plays.drop(out_shots.index)
out_actions['next_type_id'].value_counts()
out_plays['result_id'].value_counts()
season_events['time_seconds'] = season_events['minute'] * 60 + season_events['second']
season_events["x"] = season_events["x"] * 105/100
season_events["y"] = (season_events['y']) * 68/100
season_events["end_x"] = season_events['end_x'] * 105/100
season_events["end_y"] = (season_events['end_y']) * 68/100
events_spadl = pd.read_csv('ENG-Premier League_spadl_events_24.csv')
events_spadl['type_id'].value_counts()
no_event = events_spadl[events_spadl['original_event_id'].isnull()]
shots_spadl = events_spadl.loc[events_spadl['type_id'] == 11]
season_events.type.value_counts()
events_spadl['type_id'].value_counts()
events_spadl["prev_type_id"] = events_spadl.shift(1, fill_value=0)["type_id"]
events_spadl["prev_result_id"] = events_spadl.shift(1, fill_value=0)["result_id"]
events_spadl_small = events_spadl[:200000]
events_spadl_small.to_csv("small_df.csv")
small_statsbomb = df[:100000]
small_statsbomb.to_csv("small_statsbomb.csv")
intercept_unsuc = season_events[(season_events.type == 'Interception')]
dribble = events_spadl.loc[events_spadl['type_id'] == 15]
clearance = events_spadl.loc[events_spadl['type_id'] == 18]
fouls = events_spadl.loc[events_spadl['type_id'] == 8]
intercept = events_spadl.loc[events_spadl['type_id'] == 10]
tackle = events_spadl.loc[events_spadl.type_id == 9]
intercept['prev_type_id'].value_counts()
bad_pass = events_spadl.loc[(events_spadl['type_id'] == 0) & (events_spadl['result_id'] == 0)]
bad_pass['next_type_id'].value_counts()
bad_pass = bad_pass.drop(out_pass.index)
# passes out of play
out_pass = bad_pass.loc[(bad_pass['end_x'] == 0) | (bad_pass['end_x'] == 105) | (bad_pass['end_y'] == 68) | (bad_pass['end_y'] == 0)]
out_pass_2 = events_spadl.loc[(events_spadl.type_id == 0) & ((events_spadl['end_x'] == 0) | (events_spadl['end_x'] == 105) | (events_spadl['end_y'] == 68) | (events_spadl['end_y'] == 0))]

test_game = season_events[:1580]
soccerdata/data/WhoScored/events/ENG-Premier League_2425/1821049.json"
from kloppy import statsperform

dataset = statsperform.load_event(
    ma1_data="soccerdata/data/WhoScored/events/ENG-Premier League_2425/1821049.json",
    ma3_data="soccerdata/data/WhoScored/events/ENG-Premier League_2425/1821049.json",

    # Optional arguments
    coordinates="opta",    
    pitch_length=102.5,
    pitch_width=69.0,
    event_types=["pass", "shot"],
)
spadl_repres = socceraction.spadl.opta.convert_to_actions(test_game, 32)
ws = sd.WhoScored(leagues='ENG-Premier League', seasons=2024)
actions = ws.read_events(match_id=1821049)
loader = ws.read_events(output_fmt='loader')
df_players = loader.players(game_id=1821049)
df_players.to_json('1821049.json') # indent for pretty-printing
dataset = statsperform.load_event(
    ma1_data=".spyder-py3/Soccer_Analytics/1821049.json",
    ma3_data="soccerdata/data/WhoScored/events/ENG-Premier League_2425/1821049.json",
    
    # Optional arguments
    coordinates="opta",    
    pitch_length=102.5,
    pitch_width=69.0,
    event_types=["pass", "shot"],
)

actions.head()
mod = socceraction.spadl.play_left_to_right(actions, 32)

test_game['event_id'] = test_game.index
import pandas as pd

# Sample dataframe
df = pd.DataFrame({"period_name": ["FirstHalf", "SecondHalf", "PostGame", "PreMatch"]})

# Define your mapping dictionary
period_mapping = {
    "FirstHalf": 1,
    "SecondHalf": 2,
    "PostGame": 3,  # You can keep adding custom values
    "PreMatch": 4
}

# Apply the mapping
test_game["period_id"] = test_game["period"].map(period_mapping)
season_events['period'].value_counts()
test_game_spadl = socceraction.spadl.opta.convert_to_actions(test_game, 32)
season_events['time'] = season_events['minute'] * 60 + season_events['second']
cnt= mod.groupby(['game_id', 'time_seconds'])['start_x'].count()
cnt = cnt.reset_index()
unique = mod.groupby(['game_id', 'time_seconds'])['start_x'].nunique()
unique = unique.reset_index()
#check
(cnt["start_x"] == unique["start_x"]).all()
# Returns True if every element is the same, in the same order


dataset.to_df().head()
out = season_events.loc[(((season_events["end_x"] == 0) | (season_events["end_y"] == 68)) | ((season_events["end_x"] == 105) | (season_events["end_y"] == 0)))]
