# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:19:23 2025

@author: Derek
"""
import pandas as pd
import re
import textdistance
import Levenshtein
from datetime import datetime
import warnings 

def build_player_mappings(df1, df2):
    """
    Generates a standardized mapping of player names between 
    two data sources using fuzzy string matching.

    This function compares player names from two dataframes and attempts to align them,
    prioritizing exact matches. If no exact match is found, it uses the Levenshtein ratio to 
    identify the closest match based on string similarity.
    A warning is issued if the number of players differs between sources, 
    which may indicate missing or unmatched entries.

    Parameters:
    ----------
    df1 : pd.DataFrame
        The first player data source.
    df2 : pd.DataFrame
        The second player data source.

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - A DataFrame mapping player names from df1 to their best match in df2.
        - A DataFrame containing diagnostic information about the matching process,
          including unmatched players
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
                similarity_score = Levenshtein.ratio(players_1.loc[i, "player"], players_2.loc[j, "player"])
                if similarity_score > max_score:
                    max_score = similarity_score
                    best_name = players_2.loc[j, "player"]
                    best_index = j
            players_1.at[i, "name_2"] = best_name
            #debugging
            #print(players_1.loc[i, "player"] + ' ' + best_name +' '+ str(max_score))
            players_2.at[best_index, "check"] = 2
    return players_1[['player', 'name_2']], players_2[['player', 'check']]


# Validation function for dates
def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    
def floats_match(a, b):
    """
    Compare two floats for equality based on the number of decimal places
    in the float with fewer digits after the decimal point.

    Parameters:
        a (float): The first floating-point number to compare.
        b (float): The second floating-point number to compare.

    Returns:
        bool: True if the numbers are equal when rounded to the smaller number
              of decimal places; False otherwise.
    """
    s_a, s_b = str(a), str(b)
    d_a = len(s_a.split('.')[-1].rstrip('0')) if '.' in s_a else 0
    d_b = len(s_b.split('.')[-1].rstrip('0')) if '.' in s_b else 0
    decimals = min(d_a, d_b)
    return round(a, decimals) == round(b, decimals)
    
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
def prioritize_exact_match(shot_1, shot_2):
    """
    Assigns priority values to shots in the primary DataFrame (`shot_1`) based on exact matches
    with shots in the secondary DataFrame (`shot_2`).

    A shot is considered an exact match if:
        - `game_idx` and `minute` are equal in both DataFrames, and
        - `scale_x` and `scale_y` in `shot_2` match `start_x` and `start_y` in `shot_1` (using float precision comparison).

    Additionally, near matches are considered if:
        - `game_idx` is equal, and
        - `minute` differs by ±1 with `second` near the boundary (≥55 or ≤5), and
        - coordinates still match.

    Shots with exact or near matches are assigned a priority of 0.
    All other shots retain a default priority based on the length of `shot_1`.

    Parameters:
        shot_1 (pd.DataFrame): Primary DataFrame containing shots to be prioritized.
        shot_2 (pd.DataFrame): Secondary DataFrame containing reference shots for matching.

    Returns:
        pd.DataFrame: A sorted version of `shot_1` with updated 'priority' values,
                      sorted by 'game_idx', 'minute', and 'priority'.
    """
    shot_1['priority'] = len(shot_1)
    for i, row in shot_1.iterrows():
        for j, row2 in shot_2.iterrows():
            if row2["game_idx"] == row["game_idx"] and row2["minute"] == row["minute"] and (
                floats_match(row2["scale_x"], row["start_x"]) and floats_match(row2["scale_y"], row["start_y"])):
                    shot_1.at[i, "priority"] = 0
                    break
            if row2["game_idx"] == row["game_idx"] and (
                (row2["minute"] == row["minute"] + 1 and row['second'] >= 55) or
                (row2["minute"] == row["minute"] - 1 and row['second'] <= 5)
                ) and floats_match(row2["scale_x"], row["start_x"]) and floats_match(row2["scale_y"], row["start_y"]):
                    shot_1.at[i, "priority"] = 0
                    break
    return shot_1.sort_values(by=['game_idx', 'minute','second','priority'])
            
            

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
    shots_df2['check'] = 0
    for i in range(len(player_mapping)):
        shots_1 = shots_df1.loc[shots_df1.player == player_mapping.loc[i, "player"]].copy().sort_values(by=['game_idx', 'minute'])
        shots_2 = shots_df2.loc[shots_df2.player == player_mapping.loc[i, "name_2"]].copy().sort_values(by=['game_idx', 'minute'])
        #apply function to prioritize exact matches
        #shots_1 = prioritize_exact_match(shots_1, shots_2)
        shots_2['check'] = 0

        for j, row in shots_1.iterrows():
            poss_shots = shots_2.loc[
                (shots_2.game_idx == row["game_idx"]) &
                (floats_match(shots_2.scale_x, row["start_x"])) &
                (floats_match(shots_2.scale_y, row["start_y"]))
            ]
            for k, row2 in poss_shots.iterrows():
                if shots_2.at[k, "check"] == 1:
                    continue
                if ((row["second"] <= 5 and (row2["minute"] == row["minute"] or row2["minute"] == row["minute"] - 1)) or 
                    (row["second"] >= 55 and (row2["minute"] == row["minute"] or row2["minute"] == row["minute"] + 1)) or
                    (row2["minute"] == row["minute"])
                ):
                    shots_2.at[k, "check"] = 1
                    shots_df2.at[k, "check"] = shots_df2.at[k, "check"] + 1
                    shots_df1.at[j, "xg"] = row2["xg"]
                    shots_df1.at[j, "minute_2"] = row2["minute"]
                    shots_df1.at[j, "player_2"] = row2["player"]
                    shots_df1.at[j, "x_2"] = row2["scale_x"]
                    shots_df1.at[j, "y_2"] = row2["scale_y"]
                    break
        
                
        for j, row in shots_1.iterrows():
            
            check = 0
            #shot has already been matched
            if row["xg"] >= 0:
                continue
            poss_shots = shots_2.loc[
                (shots_2.game_idx == row["game_idx"]) &
                (shots_2.check == 0) &
                (shots_2.minute == row["minute"])
            ][['scale_x', 'scale_y']]
            poss_shots['dist_away'] = (poss_shots['scale_x'] - row["start_x"])**2 + (poss_shots['scale_y'] - row["start_y"])**2
            sorted_poss_shots = poss_shots.sort_values(by='dist_away')
            for idx, shot in sorted_poss_shots.iterrows():
                if shots_2.loc[idx, "check"] == 0:
                    check = 1
                    shots_2.at[idx, "check"] = 1
                    shots_df2.at[idx, "check"] = shots_df2.at[idx, "check"] + 1
                    shots_df1.at[j, "xg"] = shots_2.loc[idx, "xg"]
                    shots_df1.at[j, "minute_2"] = shots_2.loc[idx, "minute"]
                    shots_df1.at[j, "player_2"] = shots_2.loc[idx, "player"]
                    shots_df1.at[j, "x_2"] = shots_2.loc[idx, "scale_x"]
                    shots_df1.at[j, "y_2"] = shots_2.loc[idx, "scale_y"]
                    break
            if check == 0:
                
                if row["second"] >= 55:
                    poss_shots_2 = shots_2.loc[
                        (shots_2.game_idx == row["game_idx"]) &
                        (shots_2.check == 0) &
                        (shots_2.minute == row["minute"] + 1)
                    ][['scale_x', 'scale_y']]    
                elif row["second"] <= 5:
                    poss_shots_2 = shots_2.loc[
                        (shots_2.game_idx == row["game_idx"]) &
                        (shots_2.check == 0) &
                        (shots_2.minute == row["minute"] - 1)
                    ][['scale_x', 'scale_y']]   
                else:
                    continue
                
                poss_shots_2['dist_away'] = (poss_shots_2['scale_x'] - row["start_x"])**2 + (poss_shots_2['scale_y'] - row["start_y"])**2
                sorted_poss_shots = poss_shots_2.sort_values(by='dist_away')
                for idx, shot in sorted_poss_shots.iterrows():
                    if shots_2.loc[idx, "check"] == 0:
                        check = 1
                        shots_2.at[idx, "check"] = 1
                        shots_df2.at[idx, "check"] = shots_df2.at[idx, "check"] + 1
                        shots_df1.at[j, "xg"] = shots_2.loc[idx, "xg"]
                        shots_df1.at[j, "minute_2"] = shots_2.loc[idx, "minute"]
                        shots_df1.at[j, "player_2"] = shots_2.loc[idx, "player"]
                        shots_df1.at[j, "x_2"] = shots_2.loc[idx, "scale_x"]
                        shots_df1.at[j, "y_2"] = shots_2.loc[idx, "scale_y"]
                        break    
                      
                  
    return shots_df1
    '''
    shots_df1['xg'] = -1.0
    shots_df1['minute_2'] = -1
    shots_df1['player_2'] = ''
    shots_df1['x_2'] = -1.0
    shots_df1['y_2'] = -1.0
    shots_df2['check'] = 0
    for i in range(len(player_mapping)):
        shots_1 = shots_df1.loc[shots_df1.player == player_mapping.loc[i, "player"]].copy().sort_values(by=['game_idx', 'minute'])
        shots_2 = shots_df2.loc[shots_df2.player == player_mapping.loc[i, "name_2"]].copy().sort_values(by=['game_idx', 'minute'])
        #apply function to prioritize exact matches
        #shots_1 = prioritize_exact_match(shots_1, shots_2)
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
                        sorted_poss_shots = poss_shots.sort_values(by='dist_away')
                        for idx, shot in sorted_poss_shots.iterrows():
                            if shots_2.loc[idx, "check"] == 0:
                                check = 1
                                shots_2.at[idx, "check"] = 1
                                shots_df2.at[idx, "check"] = shots_df2.at[idx, "check"] + 1
                                shots_df1.at[j, "xg"] = shots_2.loc[idx, "xg"]
                                shots_df1.at[j, "minute_2"] = shots_2.loc[idx, "minute"]
                                shots_df1.at[j, "player_2"] = shots_2.loc[idx, "player"]
                                shots_df1.at[j, "x_2"] = shots_2.loc[idx, "scale_x"]
                                shots_df1.at[j, "y_2"] = shots_2.loc[idx, "scale_y"]
                                break
                        if check == 1:
                            break
                    else:
                        check = 1
                        shots_2.at[k, "check"] = 1
                        shots_df2.at[k, "check"] = shots_df2.at[k, "check"] + 1
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
                            #possible shots to match to
                            poss_shots = shots_2.loc[
                                (shots_2.game_idx == row2["game_idx"]) &
                                (shots_2.minute == row2["minute"])
                            ][['scale_x', 'scale_y']]
                            poss_shots['dist_away'] = (poss_shots['scale_x'] - row["start_x"])**2 + (poss_shots['scale_y'] - row["start_y"])**2
                            sorted_poss_shots = poss_shots.sort_values(by='dist_away')
                            for idx, shot in sorted_poss_shots.iterrows():
                                if shots_2.loc[idx, "check"] == 0:
                                    check = 1
                                    shots_2.at[idx, "check"] = 1
                                    shots_df2.at[idx, "check"] = shots_df2.at[idx, "check"] + 1
                                    shots_df1.at[j, "xg"] = shots_2.loc[idx, "xg"]
                                    shots_df1.at[j, "minute_2"] = shots_2.loc[idx, "minute"]
                                    shots_df1.at[j, "player_2"] = shots_2.loc[idx, "player"]
                                    shots_df1.at[j, "x_2"] = shots_2.loc[idx, "scale_x"]
                                    shots_df1.at[j, "y_2"] = shots_2.loc[idx, "scale_y"]
                                    break
                            if check == 1:
                                break
                        else:
                            check = 1
                            shots_2.at[k, "check"] = 1
                            shots_df2.at[k, "check"] = shots_df2.at[k, "check"] + 1
                            shots_df1.at[j, "xg"] = row2["xg"]
                            shots_df1.at[j, "minute_2"] = row2["minute"]
                            shots_df1.at[j, "player_2"] = row2["player"]
                            shots_df1.at[j, "x_2"] = row2["scale_x"]
                            shots_df1.at[j, "y_2"] = row2["scale_y"]

    return shots_df1
'''
    
    
    
