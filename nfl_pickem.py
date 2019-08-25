"""nfl_pickem.py

   Author: Adam J. Vogt
   Project Origin Date: 09/07/2018
   Modified Date: 08/24/2019

   package for selecting optimal sequence of NFL team picks
   in a survivor pool
"""

import time
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pulp as pl

class Pickem(object):
    def __init__(self,
                 file_path='../nfl-pickem/data/nfl_games.csv'):
        self.file_path = file_path
        self.team_schedule_ = None
        self.data_ = None
    
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
        df['win_prob'] = \
            1 / (np.power(10, -1*(df.elo1_week - df.elo2_week)/400) + 1)
        
        # team1 away
        cond = (df.neutral < 0.5)&(df.home < 0.5)
        df.loc[cond, 'win_prob'] = \
            1 / (np.power(10, -1*(df[cond].elo1_week - df[cond].elo2_week - 65)/400) + 1)
        
        # team1 home
        cond = (df.neutral < 0.5)&(df.home > 0.5)
        df.loc[cond, 'win_prob'] = \
            1 / (np.power(10, -1*(df[cond].elo1_week + 65 - df[cond].elo2_week)/400) + 1)
        
        return df

    def pull_data(self):

        ts = pd.read_csv(self.file_path)
        ts = self._calculate_week(ts)

        self.data_ = ts
        print('Data successfully pulled!')
        print('Seasons %i-%i: %i Games'
              %(ts.season.min(), ts.season.max(), ts.shape[0]))

    def build_schedule(self, 
                       season=2017,
                       elo_week=1):
        
        if self.data_ is None:
            return print('No game schedule data, please try pull_data() method')
        else:
            team_schedule = self.data_.copy()

        # Sort on season
        team_schedule = team_schedule[team_schedule.season == season]

        team_schedule['result2'] = 1 - team_schedule.result1.values
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
                cond = team_schedule['team%i'%i] == team
                if elo_week == 1:
                    elo = team_schedule[cond]['elo%i'%i].values[0]
                else:
                    elo = team_schedule.loc[
                        cond&(team_schedule.week <= elo_week),
                        'elo%i'%i
                    ].values[-1]
                team_schedule.loc[cond, 'elo%i_week'%i] = elo

        team_schedule['team'] = team_schedule['team1']
        team_schedule = self._calculate_probabilities(team_schedule)
        team_schedule = team_schedule.sort_values(by=['date', 'win_prob'],
                                                  ascending=[True, False])
        
        # Sort on elo_week
        team_schedule = team_schedule[team_schedule.week >= elo_week]

        return team_schedule.reset_index()

    def pick_optimization(self, df, verbose=False):
        
        prob = pl.LpProblem("pickem problem", pl.LpMaximize)
        picks = pl.LpVariable.dicts("pick", df.index.values, 0, 1, pl.LpInteger)

        # One Pick Per Week
        week_pick_mat = pd.crosstab(
            index=df.week, columns=df.index
        )

        week_map = [0] * week_pick_mat.shape[0]
        for i, j in zip(*np.asarray(week_pick_mat).nonzero()):
            week_map[i] += week_pick_mat.iloc[i, j] * picks[j]

        for i in range(week_pick_mat.shape[0]):
            prob += week_map[i] == 1

        # One Pick Per Team
        team_pick_mat = pd.crosstab(
            index=df.team1, columns=df.index
        )

        team_map = [0] * team_pick_mat.shape[0]
        for i, j in zip(*np.asarray(team_pick_mat).nonzero()):
            team_map[i] += team_pick_mat.iloc[i, j] * picks[j]

        for r in range(team_pick_mat.shape[0]):
            prob += team_map[r] <= 1

        # Add objective
        prob.objective += pl.lpSum((np.log(df.win_prob.values[j]).round(4) * picks[j] for j in range(df.shape[0])))

        prob.solve(pl.PULP_CBC_CMD(msg=(1 if verbose else 0)))

        ind = []
        for v in prob.variables():
            if v.varValue > 0:
                ind.append(int(v.name.split('_')[1]))

        return ind
    
    def compare_picks(self,
                      season=2017,
                      current_week=1,
                      max_week=17,
                      prior_picks=[]):
        ts = self.build_schedule(season=season,
                                 elo_week=current_week)
        ts = ts[~ts.team1.isin(prior_picks)].reset_index()
        
        max_week = min(ts.week.max(), max_week)
        picks = []
        for week in range(max_week, 
                          ts.week.min()-1, 
                          -1):
            ind = self.pick_optimization(
                ts[ts.week <= week]
            )
            picks.append(
                list(
                    ts.loc[ind, ['team1', 'win_prob']].apply(
                        lambda x: '%s_%.3f'%(x[0], x[1]), 
                        axis=1
                    ).values
                )
            )
        
        print('Team_WinProbability')
        for pick in picks:
            print(pick)
