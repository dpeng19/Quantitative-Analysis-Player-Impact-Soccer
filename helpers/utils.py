# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:19:23 2025

@author: Derek
"""
import pandas as pd
import re
import textdistance
from datetime import datetime
import warnings 

def get_player_mappings(df1, df2):
    """
    Creates a standardized player mapping of the player names across WhoScored and Understat.
    Uses fuzzy string matching, specifically Jaro-Winkler, to help match players 
    if no exact match is found. Warns if data sources have different number
    of players.
    

    Parameters:
    df1 (pd.DataFrame): WhoScored data
    df2 (pd.DataFrame): Understat data
    
    Returns:
    pd.DataFrame: Mapping of player names across the two sources.
    """
    players_1 = pd.DataFrame(df1['player'].drop_duplicates())
    players_2 = pd.DataFrame(df2['player'].drop_duplicates())
    if len(players_1) != len(players_2):
        warnings.warn("Number of players not equal across the dataframes.", UserWarning)
    else:
        print("Same number of players.")
    players_1 = players_1.sort_values(by = 'player').reset_index(drop = True)
    players_2 = players_2.sort_values(by = 'player').reset_index(drop=True)
    players_1['name_2'] = ''
    players_2['check'] = 0
    players_1['last_name'] = players_1['player'].apply(
        lambda x: x[x.rindex(' ') + 1:] if ' ' in x else x
    )
    players_1['first_i'] = players_1['player'].apply(
        lambda x: x[:1] if ' ' in x else x
    )
    players_2['last_name'] = players_2['player'].apply(
        lambda x: x[x.rindex(' ') + 1:] if ' ' in x else x
    )
    players_2['first_i'] = players_2['player'].apply(
        lambda x: x[:1] if ' ' in x else x
    )
    players_2['similar_score'] = 0

    #players_1.loc[374, "player"] == players_2.loc[373, "player"]
    for i in range(len(players_1)):
        for j in range(len(players_2)):
            if players_2.loc[j, "check"] == 1:
                continue
            if players_1.loc[i, "player"] == players_2.loc[j, "player"]:
                players_2.at[j, "check"] = 1
                players_1.at[i, "name_2"] = players_2.loc[j, "player"]
                break
       
        
    for i in range(len(players_1)):
        if players_1.at[i, "name_2"] == '': 
            best_name = ''
            max_score = 0
            best_index = -1
            for j in range(len(players_2)):
                if players_2.loc[j, "check"] == 1:
                    continue
                similarity_score = textdistance.jaro_winkler(players_1.loc[i, "player"], players_2.loc[j, "player"])
                if similarity_score > max_score:
                    max_score = similarity_score
                    best_name = players_2.loc[j, "player"]
                    best_index = j
            players_1.at[i, "name_2"] = best_name
            #debugging
            #print(players_1.loc[i, "player"] + ' ' + best_name +' '+ str(max_score))
            players_2.at[best_index, "check"] = 2
    return players_1[['player', 'name_2']]


# Validation function for dates
def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    
def standardized_game_idx(sch1, sch2):
    """
    Aligns two game schedules by standardizing the 'game_idx' column across both DataFrames.

    Parameters:
        sch1 (pd.DataFrame): Reference schedule containing correct 'game_idx' values.
        sch2 (pd.DataFrame): Secondary schedule to be updated with standardized 'game_idx'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated sch1 and sch2 DataFrames.
    
    """
    
    sch2['game_idx'] = -1
    
    for i in range(len(sch1)):
        check = 0
        for j in range(len(sch2)):
            if sch2.loc[j, "game_idx"] != -1:
                continue
            if sch1.loc[i, "game"] == sch2.loc[j, "game"]:
                check = 1
                sch2.at[j, "game_idx"] = sch1.loc[i, "game_idx"]
                break
        if check == 0:
            max_score = 0
            best_index = -1
            for j in range(len(sch2)):
                if sch2.loc[j, "game_idx"] != -1:
                    continue
                if sch1.loc[i, "date_only"] == sch2.loc[j, "date_only"]:
                    sim_score = textdistance.jaro_winkler(sch1.loc[i, "game"], sch2.loc[j, "game"])
                    if sim_score > max_score:
                        max_score = sim_score
                        best_index = j
                        
            sch2.at[best_index, "game_idx"] = sch1.loc[i, "game_idx"]
    return sch1, sch2

def add_expected_goals_info(shots_df1, shots_df2, player_mapping, merge_shots_group):
    """
    Matches shots from shots_df1 to shots in shots_df2 using player_mapping.
    Enriches shots_df1 with expected goal (xG) values when a match is found.

    Matching is based on:
    - Player identity (mapped via player_mapping)
    - Game index
    - Minute (exact match or ±1 with second tolerance)
    - If multiple candidates exist, the closest shot is selected using Euclidean distance between coordinates.

    While shots_df1 already contains positional data, coordinates from shots_df2 
    are added for verification purposes only.
    Shots that cannot be matched retain an xG value of -1.0.

    Parameters:
    - shots_df1 (pd.DataFrame): Primary dataset to be enriched with xG values.
    - shots_df2 (pd.DataFrame): Secondary dataset containing xG and positional data.
    - player_mapping (pd.DataFrame): Maps player names between the two datasets.
    - merge_shots_group (pd.DataFrame): Aggregated shot counts used to resolve minute-level duplicates.

    Returns:
    - pd.DataFrame: Updated shots_df1 with xG and verification data from shots_df2.
    """
    shots_df1['xg'] = -1.0
    shots_df1['minute_2'] = -1
    shots_df1['player_2'] = ''
    shots_df1['x_2'] = -1.0
    shots_df1['y_2'] = -1.0

    for i in range(len(player_mapping)):
        shots_1 = shots_df1.loc[shots_df1.player == player_mapping.loc[i, "player"]].copy().sort_values(by=['game_idx', 'minute'])
        shots_2 = shots_df2.loc[shots_df2.player == player_mapping.loc[i, "name_2"]].copy().sort_values(by=['game_idx', 'minute'])
        shots_2['check'] = 0

        for j, row in shots_1.iterrows():
            check = 0
            for k, row2 in shots_2.iterrows():
                if row2["check"] != 0:
                    continue
                if row2["game_idx"] > row["game_idx"]:
                    break
                if row2["game_idx"] == row["game_idx"] and row2["minute"] == row["minute"]:
                    if merge_shots_group.loc[
                        (merge_shots_group.game_idx == row2["game_idx"]) &
                        (merge_shots_group.player == player_mapping.loc[i, "name_2"]) &
                        (merge_shots_group.minute == row2["minute"])
                    ]['count'].values[0] > 1:
                        poss_shots = shots_2.loc[
                            (shots_2.game_idx == row2["game_idx"]) &
                            (shots_2.minute == row2["minute"])
                        ][['scale_x', 'scale_y']]
                        poss_shots['dist_away'] = (poss_shots['scale_x'] - row["start_x"])**2 + (poss_shots['scale_y'] - row["start_y"])**2
                        closest_idx = poss_shots['dist_away'].idxmin()

                        check = 1
                        shots_2.at[closest_idx, "check"] = 1
                        shots_df1.at[j, "xg"] = shots_2.loc[closest_idx, "xg"]
                        shots_df1.at[j, "minute_2"] = shots_2.loc[closest_idx, "minute"]
                        shots_df1.at[j, "player_2"] = shots_2.loc[closest_idx, "player"]
                        shots_df1.at[j, "x_2"] = shots_2.loc[closest_idx, "scale_x"]
                        shots_df1.at[j, "y_2"] = shots_2.loc[closest_idx, "scale_y"]
                        break
                    else:
                        check = 1
                        shots_2.at[k, "check"] = 1
                        shots_df1.at[j, "xg"] = row2["xg"]
                        shots_df1.at[j, "minute_2"] = row2["minute"]
                        shots_df1.at[j, "player_2"] = row2["player"]
                        shots_df1.at[j, "x_2"] = row2["scale_x"]
                        shots_df1.at[j, "y_2"] = row2["scale_y"]
                        break

            if check == 0:
                for k, row2 in shots_2.iterrows():
                    if row2["check"] != 0:
                        continue
                    if row2["game_idx"] > row["game_idx"]:
                        break
                    if row2["game_idx"] == row["game_idx"] and (
                        (row2["minute"] == row["minute"] + 1 and row['second'] >= 55) or
                        (row2["minute"] == row["minute"] - 1 and row['second'] <= 5)
                    ):
                        if merge_shots_group.loc[
                            (merge_shots_group.game_idx == row2["game_idx"]) &
                            (merge_shots_group.player == player_mapping.loc[i, "name_2"]) &
                            (merge_shots_group.minute == row2["minute"])
                        ]['count'].values[0] > 1:
                            poss_shots = shots_2.loc[
                                (shots_2.game_idx == row2["game_idx"]) &
                                (shots_2.minute == row2["minute"])
                            ][['scale_x', 'scale_y']]
                            poss_shots['dist_away'] = (poss_shots['scale_x'] - row["start_x"])**2 + (poss_shots['scale_y'] - row["start_y"])**2
                            closest_idx = poss_shots['dist_away'].idxmin()

                            check = 1
                            shots_2.at[closest_idx, "check"] = 1
                            shots_df1.at[j, "xg"] = shots_2.loc[closest_idx, "xg"]
                            shots_df1.at[j, "minute_2"] = shots_2.loc[closest_idx, "minute"]
                            shots_df1.at[j, "player_2"] = shots_2.loc[closest_idx, "player"]
                            shots_df1.at[j, "x_2"] = shots_2.loc[closest_idx, "scale_x"]
                            shots_df1.at[j, "y_2"] = shots_2.loc[closest_idx, "scale_y"]
                            break
                        else:
                            check = 1
                            shots_2.at[k, "check"] = 1
                            shots_df1.at[j, "xg"] = row2["xg"]
                            shots_df1.at[j, "minute_2"] = row2["minute"]
                            shots_df1.at[j, "player_2"] = row2["player"]
                            shots_df1.at[j, "x_2"] = row2["scale_x"]
                            shots_df1.at[j, "y_2"] = row2["scale_y"]

    return shots_df1
    
    
    
