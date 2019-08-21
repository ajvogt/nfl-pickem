"""nfl_pickem.py

   Author: Adam J. Vogt
   Project Origin Date: 09/07/2018
   Modified Date: 08/20/2019

   package for selecting optimal sequence of NFL team picks
   in a survivor pool
"""

import time
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

class Pickem(object):
    def __init__(self,
                 file_path='../nfl-pickem/data/nfl_games.csv'):
        self.file_path = file_path
    
    def _calculate_week(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['weekday'] = df.date.apply(lambda x: x.weekday())

        # adjust day of week to football week
        date_adjustments = [
            {'weekday': 0, 'days': -1},
            {'weekday': 1, 'days': -2},
            {'weekday': 2, 'days': 4},
            {'weekday': 3, 'days': 3},
            {'weekday': 4, 'days': 2},
            {'weekday': 5, 'days': 1}
        ]

        # assign adjusted weekday
        for adj in date_adjustments:
            df.loc[df.weekday == adj['weekday'], 'date'] = \
                df.loc[df.weekday == adj['weekday']].date.apply(
                    lambda x: x + datetime.timedelta(days=adj['days']))

        # assign week to each date in season
        df['week'] = 0
        for season in df.season.unique():
            weeks = np.sort(df[df.season == season].date.unique())
            for week in range(len(weeks)):
                df.loc[(df.season == season)&
                    (df.date == weeks[week]), 'week'] = week + 1

        return df

    def _calculate_probabilities(self, df):
        # neutral
        df['win_pct'] = \
            1 / (np.power(10, -1*(df.elo1_week - df.elo2_week)/400) + 1)
        
        # team1 away
        cond = (df.neutral < 0.5)&(df.home < 0.5)
        df.loc[cond, 'win_pct'] = \
            1 / (np.power(10, -1*(df[cond].elo1_week - df[cond].elo2_week - 65)/400) + 1)
        
        # team1 home
        cond = (df.neutral < 0.5)&(df.home > 0.5)
        df.loc[cond, 'win_pct'] = \
            1 / (np.power(10, -1*(df[cond].elo1_week + 65 - df[cond].elo2_week)/400) + 1)
        
        return df

    def build_schedule(self, 
                       season=2017,
                       elo_week=1):
        
        # Read in file, sort on season, add week, sort on week
        team_schedule = pd.read_csv(self.file_path)
        team_schedule = team_schedule[team_schedule.season == season]
        team_schedule = self._calculate_week(team_schedule)
        team_schedule = team_schedule[team_schedule.week >= elo_week]

        team_schedule['result2'] = 1 - team_schedule.result1 
        for col in ['result1', 'result2']:
            team_schedule.loc[team_schedule[col] < 0.9, col] = 0
            team_schedule.loc[:, col] = team_schedule[col].astype('int64')

        tmp1 = team_schedule.copy()
        tmp1['home'] = 1
        del tmp1['result2']
        tmp2 = team_schedule.copy()
        tmp2['home'] = 0
        tmp2.loc[:, 'elo_prob1'] = 1 - tmp2['elo_prob1'].values
        del tmp2['result1']
        tmp2 = tmp2.rename(columns={
            'team1': 'team2',
            'team2': 'team1',
            'elo1': 'elo2',
            'elo2': 'elo1',
            'score1': 'score2',
            'score2': 'score1',
            'result2': 'result1' 
        })
        team_schedule = pd.concat([tmp1, tmp2], sort=False)
        team_schedule = team_schedule.sort_values(by='date')

        # calculating probabilities based on elo week
        for i in [1, 2]:
            team_schedule['elo%i_week'%i] = np.nan
            for team in team_schedule['team%i'%i].unique():
                team_schedule.loc[
                    team_schedule['team%i'%i] == team, 'elo%i_week'%i
                ] = \
                    team_schedule[
                        team_schedule['team%i'%i] == team
                    ]['elo%i'%i].values[0]

        team_schedule['team'] = team_schedule['team1']
        team_schedule = self._calculate_probabilities(team_schedule)
        team_schedule = team_schedule.sort_values(by=['date', 'win_pct'],
                                                  ascending=[True, False])

        return team_schedule.reset_index()
