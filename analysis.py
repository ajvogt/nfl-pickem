import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/nfl_games.csv')
df['date'] = pd.to_datetime(df['date'])

df['weekday'] = df.date.apply(lambda x: x.weekday())

date_adjustments = [
    {'weekday': 0, 'days': -1},
    {'weekday': 1, 'days': -2},
    {'weekday': 2, 'days': 4},
    {'weekday': 3, 'days': 3},
    {'weekday': 4, 'days': 2},
    {'weekday': 5, 'days': 1}
]

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

df['elo_opp']
df['result2'] = 1 - df['result1']
df['elo_prob2'] = 1 - df['elo_prob1']

cols1 = ['date', 'season', 'neutral', 'playoff', 'team1', 'elo1',
         'elo_prob1', 'score1', 'score2', 'result1', 'weekday', 'week', 'elo2']
cols2 = ['date', 'season', 'neutral', 'playoff', 'team2', 'elo2',
         'elo_prob2', 'score2', 'score1', 'result2', 'weekday', 'week', 'elo1']
tmp1 = df[cols1].rename(columns={'elo2': 'elo_opp'})
tmp1['home'] = True
tmp2 = df[cols2].rename(
    columns={'elo1': 'elo_opp',
             'team2': 'team1',
             'elo2': 'elo1',
             'elo_prob2': 'elo_prob1',
             'score2': 'score1',
             'score1': 'score2',
             'result2': 'result1'}
)
tmp2['home'] = False
team_schedule = pd.concat([tmp1, tmp2])

team_schedule['week-1_elo'] = np.nan

cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)&\
       (team_schedule.week) > 5

for team in team_schedule[cond].team1.unique():
    team_schedule.loc[(team_schedule.team1 == team), 
                      'week-1_elo'].iloc[1:] = \
        team_schedule[(team_schedule.team1 == team)].elo1.iloc[:-1]

team_schedule.loc[(team_schedule.team1 == team), 
                      'week-1_elo'] = \
        team_schedule[(team_schedule.team1 == team)].elo1.iloc[:-1]

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

plt.plot(bins[:bins.shape[0]-1], vals,
         label='ELO Model')

lr = LinearRegression()
lr.fit(bins[:bins.shape[0]-1].reshape(-1, 1), 
       np.asarray(vals))
plt.plot(bins[:bins.shape[0]-1],
         lr.predict(bins[:bins.shape[0]-1].reshape(-1, 1)),
         color='k', linestyle='--',
         label='Best Fit (%.3fx+%.3f)'%(lr.coef_, lr.intercept_))
plt.xlabel('ELO Win Probability')
plt.ylabel('Frequency of Actual Wins')
plt.legend(loc='upper left')
plt.show()

cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)

team_schedule.elo1 - team_schedule 


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
