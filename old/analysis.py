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

df['result2'] = 1 - df['result1']
df['elo_prob2'] = 1 - df['elo_prob1']

cols1 = ['date', 'season', 'neutral', 'playoff', 'team1', 'elo1', 'team2',
         'elo_prob1', 'score1', 'score2', 'result1', 'weekday', 'week', 'elo2']
cols2 = ['date', 'season', 'neutral', 'playoff', 'team2', 'elo2', 'team1',
         'elo_prob2', 'score2', 'score1', 'result2', 'weekday', 'week', 'elo1']
tmp1 = df[cols1]
tmp1['home'] = True
tmp2 = df[cols2].rename(
    columns={'team2': 'team1',
             'team1': 'team2',
             'elo2': 'elo1',
             'elo1': 'elo2',
             'elo_prob2': 'elo_prob1',
             'score2': 'score1',
             'score1': 'score2',
             'result2': 'result1'}
)
tmp2['home'] = False
team_schedule = pd.concat([tmp1, tmp2])
team_schedule = team_schedule.sort_values(by='date').reset_index()

for i in range(1,7):
    team_schedule['week-%i_elo'%i] = np.nan
    team_schedule['week-%i_elo_opp'%i] = np.nan
    cond = (team_schedule.season.isin(range(2007, 2018)))
    for team in team_schedule[cond].team1.unique():
        ind = team_schedule[team_schedule.team1 == team].index
        ind = np.concatenate((np.asarray([ind[1]]*i), ind[:-i]))
        team_schedule.loc[(team_schedule.team1 == team), 'week-%i_elo'%i] = team_schedule.elo1.iloc[ind].values
    for team in team_schedule[cond].team2.unique():
        ind = team_schedule[team_schedule.team2 == team].index
        ind = np.concatenate((np.asarray([ind[1]]*i), ind[:-i]))
        team_schedule.loc[(team_schedule.team2 == team), 'week-%i_elo_opp'%i] = team_schedule.elo1.iloc[ind].values
    team_schedule['week-%i_elo_prob'%i] = \
        1/(np.power(10, -1/400*(team_schedule['week-%i_elo'%i] - \
                        team_schedule['week-%i_elo_opp'%i]))+1)
    cond = team_schedule.home&\
        (team_schedule.neutral < 0.1)
    team_schedule.loc[cond, 'week-%i_elo_prob'%i] = \
        1/(np.power(10, -1/400*(team_schedule[cond]['week-%i_elo'%i]+65 - \
                        team_schedule[cond]['week-%i_elo_opp'%i]))+1)
    cond = ~team_schedule.home&\
        (team_schedule.neutral < 0.1)
    team_schedule.loc[cond, 'week-%i_elo_prob'%i] = \
        1/(np.power(10, -1/400*(team_schedule[cond]['week-%i_elo'%i] - \
                        team_schedule[cond]['week-%i_elo_opp'%i]-65))+1)

cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)&\
       (team_schedule.week) > 7)
for col in ['elo_prob1', 'week-1_elo_prob', 'week-2_elo_prob', 'week-3_elo_prob', 'week-4_elo_prob']:
    vals = []
    bins = np.arange(0, .99, 0.05)
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(bins[i])
        all_teams = team_schedule[
            cond&
            (team_schedule[col] >= bins[i])&
            (team_schedule[col] < bins[i+1])
        ].shape[0]
        winning_teams = team_schedule[
            cond&
            (team_schedule[col] >= bins[i])&
            (team_schedule[col] < bins[i+1])&
            (team_schedule.result1 > 0.9)
        ].shape[0]
        if all_teams > 0:
            vals.append(winning_teams / all_teams)
        else:
            vals.append(0)
    plt.plot(bins[:bins.shape[0]-1], vals,
             label='ELO Model %s'%col)
    lr = LinearRegression()
    lr.fit(bins[:bins.shape[0]-1].reshape(-1, 1), 
           np.asarray(vals))
    plt.plot(bins[:bins.shape[0]-1],
            lr.predict(bins[:bins.shape[0]-1].reshape(-1, 1)),
            color='k', linestyle='--',
            label='Best Fit %s (%.3fx+%.3f)'%(col, lr.coef_, lr.intercept_))

plt.xlabel('ELO Win Probability')
plt.ylabel('Frequency of Actual Wins')
plt.legend(loc='upper left')
plt.show()

# Calculating log-loss
cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)&\
       (team_schedule.week > 9)&\
       (team_schedule.home)# &\
       # (team_schedule.elo_prob1 > 0.7)
ts = team_schedule[cond].copy()
tempering = {'elo_prob1': 0, 
             'week-1_elo_prob': 1-(119-115)/1, 
             'week-2_elo_prob': 1/np.exp(3.68), 
             'week-3_elo_prob': 1/np.exp(1.63), 
             'week-4_elo_prob': 1/np.exp(6.12),
             'week-5_elo_prob': 0,
             'week-6_elo_prob': 0}
for key in tempering.keys():
    probs = (ts[key].values - 0.5)*tempering[key]+0.5
    original = (-1*ts.result1*np.log(ts[key])+(1-ts.result1)*np.log(1-ts[key])).sum()
    tempered = (-1*ts.result1*np.log(probs)+(1-ts.result1)*np.log(1-probs)).sum()
    # import pdb; pdb.set_trace()
    print('%s: %.4f vs. %.4f'%(key, original, tempered))

cols = ['week', 'team1', 'team2', 'result1'] + list(tempering.keys())
ts[cols].head(20)

for i in range(1, 7):
    print('prob change: %.5f (+/-%.5f)'
          %((ts.elo_prob1 - ts['week-%i_elo_prob'%i]).sum()/ts.shape[0],
            np.sqrt(np.square(ts.elo_prob1 - ts['week-%i_elo_prob'%i]).sum())/ts.shape[0]))

for i in range(1, 7):
    pd.DataFrame(ts.elo_prob1 - ts['week-%i_elo_prob'%i]).hist(label='%i'%i)

plt.legend()
plt.show()

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
