import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.read_csv('data/nfl_games.csv')
df['date'] = pd.to_datetime(df['date'])

df['weekday'] = df.date.apply(lambda x: x.weekday())

date_adjustments = [{'weekday': 0, 'days': -1},
                    {'weekday': 1, 'days': -2},
                    {'weekday': 2, 'days': 4},
                    {'weekday': 3, 'days': 3},
                    {'weekday': 4, 'days': 2},
                    {'weekday': 5, 'days': 1}]

for adj in date_adjustments:
    df.loc[df.weekday == adj['weekday'], 'date'] = \
        df.loc[df.weekday == adj['weekday']].date.apply(
            lambda x: x + datetime.timedelta(days=adj['days']))

df['week'] = 0
for season in df.season.unique():
    weeks = np.sort(df[df.season == season].date.unique())
    for week in range(len(weeks)):
        df.loc[(df.season == season)&
               (df.date == weeks[week]), 'week'] = week + 1

df['result2'] = 1 - df['result1']
df['elo_prob2'] = 1 - df['elo_prob1']

cols1 = ['date', 'season', 'neutral', 'playoff', 'team1', 'elo1',
         'elo_prob1', 'score1', 'score2', 'result1', 'weekday', 'week']
cols2 = ['date', 'season', 'neutral', 'playoff', 'team2', 'elo2',
         'elo_prob2', 'score2', 'score1', 'result2', 'weekday', 'week']
team_schedule = pd.concat([df[cols1],
                           df[cols2].rename(columns={'team2': 'team1',
                                                     'elo2': 'elo1',
                                                     'elo_prob2': 'elo_prob1',
                                                     'score2': 'score1',
                                                     'score1': 'score2',
                                                     'result2': 'result1'})])
    
cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)
vals = []
bins = np.arange(0, .99, 0.025)
bin_labels = []
for i in range(len(bins)-1):
    bin_labels.append(bins[i])
    all_teams = team_schedule[
        cond&
        (team_schedule.elo_prob1 >= bins[i])&
        (team_schedule.elo_prob1 < bins[i+1])
    ].shape[0]
    winning_teams = team_schedule[
        cond&
        (team_schedule.elo_prob1 >= bins[i])&
        (team_schedule.elo_prob1 < bins[i+1])&
        (team_schedule.result1 > 0.9)
    ].shape[0]
    if all_teams > 0:
        vals.append(winning_teams / all_teams)
    else:
        vals.append(0)

plt.plot(bins[:bins.shape[0]-1], vals)
plt.xlabel('ELO Win Probability')
plt.ylabel('Frequency of Actual Wins')
plt.show()

cond = (team_schedule.season == 2017)&\
       (team_schedule.playoff == 0)
team_sch = team_schedule[cond]
elo_week = pd.crosstab(index=team_sch.team1,
                       columns=team_sch.week,
                       values=team_sch.elo1,
                       aggfunc=np.nanmean)
elo_week.columns = elo_week.columns.astype('str')


elo_week.loc[elo_week['1'].isnull(), '1'] = \
    elo_week.loc[elo_week['1'].isnull(), '2']

for col in elo_week.columns[1:]:
    elo_week.loc[elo_week[col].isnull(), col] = \
        elo_week.loc[elo_week[col].isnull(), '%i'%(int(col)-1)]
