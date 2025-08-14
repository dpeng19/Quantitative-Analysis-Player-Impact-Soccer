# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:40:10 2025

@author: Derek
"""

import pandas as pd


# Define shot types
shot_ids = [11, 12, 13]

def isolate_chains(df):
    """
    Parameters
    ----------
    df : dataframe
        dataframe with soccer event data.

    Returns
    -------
    df: dataframe
        dataframe with isolated possesion chains

    """
    #potential +0s
    chain_team = df.iloc[0]["team_id"]
    period = df.iloc[0]["period_id"]
    stop_criterion = 0
    chain = 0
    df["possesion_chain"] = 0
    df["possesion_chain_team"] = 0
    count = 0;
    for i, row in df.iterrows():
        #add value
        df.at[i, "possesion_chain"] = chain
        df.at[i, "possesion_chain_team"] = chain_team
        #if shot, offside, or foul, add 2 to stop criteriom
        if row["type_id"] in shot_ids or row["result_id"] == 2 or row["type_id"] == 8:
                stop_criterion += 2
        
        #if ball out of field, add 2
        if row["kicked_out"] == 1:
                stop_criterion += 2
        #maybe
        #if row['team_id'] != row['next_team_id'] & row['next_type_id] != 8
        #maybe also for interception/tackle change possession if successful, even if team after  defensive action is the same
        #should add also if not period end (&row["period_id"] == period)
        #like so: if row['team_id'] != row['next_team_id'] & row["period_id"] == period:
        #why adding 2 to possesion chain value?
        if (row['team_id'] != row['next_team_id']) & (row["next_period_id"] == period):
            stop_criterion += 2
        #criterion for stopping when half ended
        if row["period_id"] != period:
                chain += 1
                if row['team_id'] != row['next_team_id']:
                    stop_criterion += 2;
                else:
                  stop_criterion = 0
                chain_team = row['team_id']
                period = row["period_id"]
                df.at[i, "possesion_chain"] = chain
                df.at[i, "possesion_chain_team"] = chain_team
        #possesion chain ended
        if stop_criterion >= 2:
            chain += 1
            stop_criterion = 0
            chain_team = row['next_team_id']
        count = count + 1
        print(count)
    return df