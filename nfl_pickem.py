"""nfl_pickem.py

   Author: Adam J. Vogt
   Project Origin Date: 09/07/2018
   Modified Date: 08/24/2019

   package for selecting optimal sequence of NFL team picks
   in a survivor pool
"""

import time
import datetime
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pulp as pl

class Pickem(object):
    def __init__(self,
                 file_path='../nfl-pickem/data/nfl_elo.csv'):
        self.file_path = file_path
        self.data_url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
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
            1 / (np.power(10, -1*(df.elo1_week - df.elo2_week + df.elo_adj)/400) + 1)
        
        return df

    def pull_data(self, auto_update=True):

        if auto_update:
            print('Pulling latest data from: %s'%self.data_url)
            new_file = requests.get(self.data_url)
            print('Saving latest data to: %s'%self.file_path)
            open(self.file_path, 'wb').write(new_file.content)

        ts = pd.read_csv(self.file_path)
        ts = self._calculate_week(ts)

        self.data_ = ts
        print('Data successfully pulled!')
        print('Seasons %i-%i: %i Games'
              %(ts.season.min(), ts.season.max(), ts.shape[0]))

    def build_schedule(self, 
                       season=2017,
                       elo_week=1,
                       qb_elo_model=False):
        
        if self.data_ is None:
            return print('No game schedule data, please try pull_data() method')
        else:
            team_schedule = self.data_.copy()

        # Sort on season
        team_schedule = team_schedule[team_schedule.season == season]

        if 'result1' not in team_schedule.columns:
            team_schedule['result1'] = 0
            cond_win = team_schedule.score1 > team_schedule.score2
            team_schedule.loc[cond_win, 'result1'] = 1
    
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
        tmp2.loc[:, 'qbelo_prob1'] = 1 - tmp2['qbelo_prob1'].values
        del tmp2['result1']
        tmp2 = tmp2.rename(columns={
            'team1': 'team2',
            'team2': 'team1',
            'elo1_pre': 'elo2_pre',
            'elo2_pre': 'elo1_pre',
            'qbelo1_pre': 'qbelo2_pre',
            'qbelo2_pre': 'qbelo1_pre',
            'qb1_adj': 'qb2_adj',
            'qb2_adj': 'qb1_adj',
            'score1': 'score2',
            'score2': 'score1',
            'result2': 'result1' 
        })
        team_schedule = pd.concat([tmp1, tmp2], sort=False)
        team_schedule = team_schedule.sort_values(by='date')
        
        # choosing elo model and finding elo home/away adjustment
        if qb_elo_model:
            for i in [1, 2]:
                team_schedule['elo%i'%i] = team_schedule['qbelo%i_pre'%i].values + \
                                           team_schedule['qb%i_adj'%i].values
            team_schedule['elo_adj'] = \
                -400*np.log10(1/team_schedule.qbelo_prob1.values - 1) - \
                (team_schedule['qbelo1_pre'].values - \
                 team_schedule['qbelo2_pre'].values + \
                 team_schedule['qb1_adj'].values - \
                 team_schedule['qb2_adj'].values)
        else:
            for i in [1, 2]:
                team_schedule['elo%i'%i] = team_schedule['elo%i_pre'%i].values
            team_schedule['elo_adj'] = \
                -400*np.log10(1/team_schedule.elo_prob1.values - 1) - \
                (team_schedule['elo1_pre'].values - \
                 team_schedule['elo2_pre'].values)

        # calculating probabilities based on elo week
        for i in [1, 2]:
            team_schedule['elo%i_week'%i] = np.nan
            for team in team_schedule[team_schedule['team%i'%i].notnull()]['team%i'%i].unique():
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

        return np.sort(ind)
    
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

    def _make_pivot_table(self, df):
        df = df[df.team1.notnull()&df.team2.notnull()]
        results = pd.DataFrame(
            np.arange(0, np.unique(np.concatenate((df.team1, df.team2))).shape[0])
        )
        results.columns = ['index']
        for week in df.week.unique():
            tmp = df[df.week == week].copy()
            tmp = tmp.sort_values(by='win_prob', ascending=False).reset_index()[['team1', 'win_prob']]
            tmp.loc[:, 'team1'] = tmp.apply(lambda x: '%s: %.3f'%(x.team1, x.win_prob), axis=1)
            results = results.join(tmp[['team1']], how='left', rsuffix='_week%s'%week)
            results = results.rename(columns={'team1': 'week%s'%week})
        del results['index']
        return results

    def compare_picks_new(self,
                      season=2017,
                      current_week=1,
                      max_week=17,
                      prior_picks=[]):
        ts = self.build_schedule(season=season,
                                 elo_week=current_week)
        results = self._make_pivot_table(ts)
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
                    ts.loc[ind, 'team1'].values
                )
            )
        
        results_array = results.values
        
        print('Team_WinProbability')
        fig, ax = plt.subplots(figsize=(15, 7.5))
        results_map = np.zeros(results.shape)
        picks.reverse()
        for pick in picks:
            print(pick)
            for i in range(len(pick)):
                cond = results[results.columns[i]].str.contains(pick[i])&\
                       results[results.columns[i]].notnull()
                # results.loc[cond, results.columns[i]] = \
                #     results.loc[cond, results.columns[i]] + '_X'
                if i < (len(pick) - 1):
                    if results_map[results.loc[cond, :].index[0], i] == 0:
                        results_map[results.loc[cond, :].index[0], i] = i
                else:
                    results_map[results.loc[cond, :].index[0], i] = 1


        print(np.unique(results_map))
        cmaps = ['Greens', 'Blues', 'Oranges', 
                 'Greys', 'Reds', 'PuRd']
        cax = ax.matshow(results_map, 
                         cmap='tab20b', aspect=0.15, alpha=0.5,
                         label='Forecast %i weeks'%int(i))
        ax.set_xticklabels(['']+list(results.columns))
        ax.set_yticklabels([])
        ax.grid(False)
        fig.colorbar(cax)
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                ax.text(x=j, y=i,
                        s=results.iloc[i, j],
                        va='center', ha='center',
                        fontsize=9)
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()
        import pdb; pdb.set_trace()
