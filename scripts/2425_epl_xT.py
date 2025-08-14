# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:30:11 2025

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
from mplsoccer import Sbopen
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binned_statistic_2d

import sys
import os
import importlib

# Build path to 'helpers' directory
cwd = os.getcwd()
helpers_path = os.path.join(cwd, 'helpers')

# Add to sys.path so Python can find your modules
sys.path.append(helpers_path)

#Import helpers
import scrape_event_data

#reload when necessary
importlib.reload(scrape_event_data)

#24/25 epl event data
season_events = scrape_event_data.get_event_data(
    leagues='ENG-Premier League', 
    seasons=2024, 
)
#all passes for the season
pass_df = season_events.loc[season_events['type'] == 'Pass']

#take only successful passes for xT calculation
move_df = pass_df.loc[pass_df['outcome_type'] == 'Successful']
#Convert to 105/68 field
move_df["x"] = move_df["x"] * 105/100
move_df["y"] = (move_df['y']) * 68/100
move_df["end_x"] = move_df['end_x'] * 105/100
move_df["end_y"] = (move_df['end_y']) * 68/100

pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
move = pitch.bin_statistic(move_df.x, move_df.y, statistic='count', bins=(16, 12), normalize=False)

fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pcm  = pitch.heatmap(move, cmap='Blues', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Moving actions 2D histogram', fontsize = 30)
plt.show()

move_count = move["statistic"]
#get the array

#goals_check.dtypes
#import json
shots = season_events[(season_events['type'] == 'Goal') | (season_events['type'].str.contains('Shot')) ]
#shots_check = season_events[(season_events['x'] == 88.5) & (season_events['y'] == 50 ) & ((season_events['type'] == 'Goal') | (season_events['type'].str.contains('Shot'))) ]

shots['is_penalty'] = shots['qualifiers'].apply(
    lambda x: 1 if any(q.get('type') == {'displayName': 'Penalty', 'value': 9} 
    for q in json.loads(x.replace("'", '"'))) else 0
)
penalty_shots= shots.loc[shots['is_penalty'] == 1]
shots['own_goal'] = shots['qualifiers'].apply(
    lambda x: 1 if any(q.get('type') == {'displayName': 'OwnGoal', 'value': 28}
    for q in json.loads(x.replace("'", '"'))) else 0
)
own_goals = shots.loc[shots['own_goal'] == 1]
#get non-penalty shots and shots that result in own goals
shot_df = shots.drop(penalty_shots.index)
shot_df = shot_df.drop(own_goals.index)
shot_df["x"] = shot_df['x'] * 105/100
shot_df["y"] = (shot_df['y']) * 68/100

#create 2D histogram of these
shot = pitch.bin_statistic(shot_df.x, shot_df.y, statistic='count', bins=(16, 12), normalize=False)

fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pcm  = pitch.heatmap(shot, cmap='Greens', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Shots 2D histogram', fontsize = 30)
plt.show()

shot_count = shot["statistic"]

goal_df = shot_df.loc[shot_df['type'] == 'Goal']


goal = pitch.bin_statistic(goal_df.x, goal_df.y, statistic = 'count', bins = (16, 12), normalize = False)
goal_count = goal["statistic"]

fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pcm  = pitch.heatmap(goal, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Goal 2D histogram', fontsize = 30)
plt.show()

# ----move probability heat map ----
move_prob = (move_count) / (move_count + shot_count)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
move["statistic"] = move_prob
pcm  = pitch.heatmap(move, cmap='Blues', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Move probability 2D histogram', fontsize = 30)
plt.show()


# ----shot probability heat map ----
shot_prob = (shot_count) / (move_count + shot_count)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
shot["statistic"] = shot_prob
pcm  = pitch.heatmap(shot, cmap='Greens', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Shot probability 2D histogram', fontsize = 30)
plt.show()

# -----goal probability heat map----
goal_prob = goal_count/shot_count
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
goal_prob[np.isnan(goal_prob)] = 0
goal["statistic"] = goal_prob
pcm  = pitch.heatmap(goal, cmap='Greens', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Goal probability 2D histogram', fontsize = 30)
plt.show()

move_df["start_sector"] = move_df.apply(lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.x), np.ravel(row.y), 
                                                               values = "None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)
#move end index
move_df["end_sector"] = move_df.apply(lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.end_x), np.ravel(row.end_y), 
                                                               values = "None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)
move_df['id'] = move_df.index
df_count_starts = move_df.groupby(["start_sector"])["id"].count().reset_index()
df_count_starts.rename(columns = {'id':'count_starts'}, inplace=True)

transition_matrices = []
for i, row in df_count_starts.iterrows():
    start_sector = row['start_sector']
    count_starts = row['count_starts']
    #get all events that started in this sector
    this_sector = move_df.loc[move_df["start_sector"] == start_sector]
    this_sector['id'] = this_sector.index
    df_cound_ends = this_sector.groupby(["end_sector"])["id"].count().reset_index()
    df_cound_ends.rename(columns = {'id':'count_ends'}, inplace=True)
    T_matrix = np.zeros((12, 16))
    for j, row2 in df_cound_ends.iterrows():
        end_sector = row2["end_sector"]
        value = row2["count_ends"]
        T_matrix[12 - end_sector[1]][end_sector[0] - 1] = value
    T_matrix = T_matrix / count_starts
    transition_matrices.append(T_matrix)
    
    
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

#Change the index here to change the zone.
goal["statistic"] = transition_matrices[90]
pcm  = pitch.heatmap(goal, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Transition probability for one of the middle zones', fontsize = 30)
plt.show()

#formula
#df_count_starts['pos'] = (df_count_starts['start_sector'][0] - 1) * 12 + df_count_starts['start_sector'][1] - 1
transition_matrices_array = np.array(transition_matrices)
xT = np.zeros((12, 16))
xT_new = np.zeros((12, 16))
for i in range(5):
    for r in range(12):
        for c in range(16):
            shoot_expected_payoff = goal_prob[r][c]*shot_prob[r][c]
            move_payoff = 0
            for r2 in range(12):
                for c2 in range(16):
                    convert_r = c + 1
                    convert_c = 12 - r
                    move_payoff = move_payoff + (transition_matrices_array[(convert_r - 1) * 12 + convert_c - 1][r2][c2] * xT[r2][c2])
                    print(move_payoff)
            #print(move_probability[r][c])
            move_expected_payoff = move_prob[r][c] * move_payoff
            xT_new[r][c] = shoot_expected_payoff + move_expected_payoff   
    xT = xT_new                                      

    #let's plot it!
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.01, title_space=0, endnote_space=0)
    goal["statistic"] = xT
    pcm  = pitch.heatmap(goal, cmap='Oranges', edgecolor='grey', ax=ax['pitch'])
    labels = pitch.label_heatmap(goal, color='blue', fontsize=9,
                             ax=ax['pitch'], ha='center', va='center', str_format="{0:,.2f}", zorder = 3)
    #legend to our plot
    ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
    cbar = plt.colorbar(pcm, cax=ax_cbar)
    txt = 'Expected Threat matrix after ' +  str(i+1) + ' moves'
    fig.suptitle(txt, fontsize = 30)
    plt.show()


#calculate XT added for each move
move_df["xT_added"] = move_df.apply(lambda row: xT[12 - row.end_sector[1]][row.end_sector[0] - 1] 
                                                      - xT[12 - row.start_sector[1]][row.start_sector[0] - 1], axis = 1)
move_df = move_df.loc[move_df['xT_added'] > 0]
#group by player
xT_by_player = move_df.groupby(["player"])["xT_added"].sum().reset_index()
final = xT_by_player.sort_values(by='xT_added', ascending=False)
#final.to_csv('GER-Bundesliga_24_topxT.csv', index=False)

