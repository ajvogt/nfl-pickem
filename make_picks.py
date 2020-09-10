"""python script for comparing different pick 
   forecasts.

Parameters
----------
season : int (year)
    the season for which the picks are being made

current_week : int [1-17]
    the week of the season from which to forecast

prior_picks : list of str
    list of team acronyms of picks in prior weeks so
    that they excluded from forecast

Returns
-------
teams : printed list
    list of team acronyms in the current season

picks : printed sequence of lists
    prints a sequency of lists forecasting from the current
    week to each of the number of weeks remaining in the
    season (e.g. forecast just this week, forecast next
    five weeks).  Elements in list contain the suggested team
    to pick and the estimated probability of them winning in
    that week if current ELO ratings are used.

Example
-------
season=2017
current_week=9
prior_picks=['NE', 'NO', 'SEA', 'BAL', 'LAR',
             'DEN', 'JAX', 'PIT']
"""
season=2020
current_week=1
prior_picks=[]
auto_update = True # will automatically pull the latest data from FiveThirtyEight
visualize_results = False # will produce a plot of the different forecasting selections

import numpy as np
import pandas as pd
from nfl_pickem import Pickem

if __name__ == "__main__":
    pk = Pickem()
    pk.pull_data(auto_update=auto_update)
    df = pk.build_schedule(season=season)
    teams = np.unique(np.concatenate(
        (df[df.team1.notnull()].team1, 
         df[df.team2.notnull()].team2)
    ))
    print('Teams previously chosen: %s'%prior_picks)
    print('Teams available to choose from: %s'
          %[x for x in teams if x not in prior_picks])
    pk.compare_picks(season=season,
                     current_week=current_week,
                     prior_picks=prior_picks,
                     visualize_results=visualize_results)
