# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 15:44:31 2025

@author: Derek
"""

import soccerdata as sd
import pandas as pd
import socceraction
import os



def get_event_data(leagues, seasons, output_fmt='events', resave = False):
    """
    Retrives all the events for the given league, season in the format specified, from
    WhoScored. Saves the dataframe as a csv file, and returns the dataframe. If 
    file already exists in the right format for the right season and league, 
    if resave is false, then it will just read from the file and return
    the dataframe. 

    Parameters:
    leagues (string or iterable, optional): IDs of Leagues to include.
    seasons (string, int or list, optional): Seasons to include. Supports multiple formats. Examples: ‘16-17’; 2016; ‘2016-17’; [14, 15, 16]
    output_fmt (str, default: 'events'): Format of the events. Can be either 'events', 'spadl', 'atomic_spadl'
    resave (bool, default: False): if file already exists, whether to retrive all the 
    events and save as csv again
    
    Returns:
    pd.DataFrame: DataFrame of all the events for specified arguements passed in
    """
    if os.path.exists('data/' + str(leagues) + '_' + output_fmt + '_' + str(seasons) + '.csv') and resave == False:
        season_events = pd.read_csv('data/' + str(leagues) + '_' + output_fmt + '_' + str(seasons) + '.csv')
        return season_events
    ws = sd.WhoScored(leagues=leagues, seasons=seasons)
    league_schedule = ws.read_schedule()
    match_ids = league_schedule['game_id'].tolist()
    season_events = pd.DataFrame()
    game_count = 0
    for match_idx in match_ids:
        if output_fmt == 'spadl':
            events = ws.read_events(match_id = match_idx, output_fmt=output_fmt)
            #event_id_start += events.shape[0]
            #events['event_id'] = event_id_start + events.index
            events = socceraction.spadl.play_left_to_right(events, league_schedule.iloc[game_count, league_schedule.columns.get_loc("home_team_id")])
        elif output_fmt == 'atomic_spadl':
            events = ws.read_events(match_id = match_idx, output_fmt=output_fmt)
            events = socceraction.atomic.spadl.play_left_to_right(events, league_schedule.iloc[game_count, league_schedule.columns.get_loc("home_team_id")])
        else:
            events = ws.read_events(match_id = match_idx)
            events = events.reset_index(drop=True)
            events['event_id'] = events.index
        season_events = pd.concat([season_events, events])
        game_count = game_count + 1
        print(game_count)
    season_events.to_csv('data/' + str(leagues) + '_' + output_fmt + '_' + str(seasons) + '.csv', index=False)
    return season_events