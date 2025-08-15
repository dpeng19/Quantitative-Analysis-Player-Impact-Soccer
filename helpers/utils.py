# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:19:23 2025

@author: Derek
"""
import pandas as pd
import re
import textdistance
from datetime import datetime

def get_player_mappings(df1, df2):
    """
    Creates a standardized player mapping of the player names across WhoScored and Understat.
    Uses fuzzy string matching, specifically Jaro-Winkler, to help match players 
    if no exact match is found. 
    

    Parameters:
    df1 (pd.DataFrame): WhoScored data
    df2 (pd.DataFrame): Understat data
    
    Returns:
    pd.DataFrame: Mapping of player names across the two sources.
    """
    players_1 = pd.DataFrame(df1['player'].drop_duplicates())
    players_2 = pd.DataFrame(df2['player'].drop_duplicates())
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
    
def get_standardized_game_idx(sch1, sch2):
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
    
