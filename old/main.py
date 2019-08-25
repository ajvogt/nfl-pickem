"""
NFL Pick-em
Author: Adam J. Vogt
Date: 09/07/2018
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

plt.style.use('fivethirtyeight')

def build_schedule(in_schedule='data/team_schedule.csv',
                   in_elo='data/team_elo.csv',
                   week='Week_1'):
    """
    Creates week-by-week schedule with each team's probability of winning
    based on elo scores"
    """
    
    team_schedule = pd.read_csv(in_schedule)
    team_elo = pd.read_csv(in_elo)
    team_elo = team_elo[['Team', week]]
    
    assert(~any(~(team_schedule.Away.isin(team_elo.Team))))
    assert(~any(~(team_schedule.Home.isin(team_elo.Team))))
    
    full_schedule = team_schedule.join(
            team_elo.rename(columns={'Team': 'Away'}).set_index('Away'), 
            on='Away'
            )
    full_schedule = full_schedule.rename(columns={week: 'ELO_away'})
    full_schedule = full_schedule.join(
            team_elo.rename(columns={'Team': 'Home'}).set_index('Home'), 
            on='Home'
            )
    full_schedule = full_schedule.rename(columns={week: 'ELO_home'})
    
    # Calculating Win Probabilities
    full_schedule.loc[full_schedule.host != 'international', 'ELO_home'] += 65
    full_schedule['win_home'] = \
        1 / (10**(-(full_schedule.ELO_home-full_schedule.ELO_away)/400) + 1)
    full_schedule['win_away'] = \
        1 / (10**(-(full_schedule.ELO_away-full_schedule.ELO_home)/400) + 1)
    
    # Combining to single team list
    away_schedule = full_schedule[['Week', 'Away', 'ELO_away', 
                                   'win_away', 'Away_Win']]
    away_schedule = away_schedule.rename(columns={'Away': 'Team',
                                                  'ELO_away': 'ELO',
                                                  'win_away': 'win_pct',
                                                  'Away_Win': 'win'})
    home_schedule = full_schedule[['Week', 'Home', 'ELO_home', 
                                   'win_home', 'Home_Win']]
    home_schedule = home_schedule.rename(columns={'Home': 'Team',
                                                  'ELO_home': 'ELO',
                                                  'win_home': 'win_pct',
                                                  'Home_Win': 'win'})
    team_schedule = pd.concat([away_schedule, home_schedule])
    
    return team_schedule


def weekly_pick(team_schedule, week, max_week, picked_teams, probs, results,
                start=None):
    if week > max_week:
        results.append({'picked_teams': picked_teams,
                                'probs': probs})
        return results
    else:
        cond = (team_schedule.Week == week) & \
               ~(team_schedule.Team.isin(picked_teams))
        for team in team_schedule[cond].Team.unique():
            # print('Week: %i, Team: %s' %(week, team))
            picked_teams_tmp = picked_teams[:]
            picked_teams_tmp.append(team)
            probs_tmp = probs[:]
            probs_tmp.append(team_schedule[cond&(team_schedule.Team == team)].win_pct.values[0])
            week_tmp = week + 1
            results = weekly_pick(team_schedule=team_schedule,
                                  week=week_tmp, max_week=max_week,
                                  picked_teams=picked_teams_tmp,
                                  probs=probs_tmp,
                                  results=results)

        return results


def main(week_min=1, week_max=17, prob_min=0,
         exclusions=[], elo_week='Week_1', verbose=False):
    """
    main function pulling in data, running all combinations,
    and returning results table
    """
    # Getting Data
    start = time.time()
    team_schedule = build_schedule(week=elo_week)
    if verbose:
        print('Build Schedule: %fs' %(time.time()-start))
    
    # Subsetting to improve performance
    team_schedule = team_schedule[(team_schedule.Week >= week_min) &
                                  (team_schedule.win_pct >= prob_min) &
                                  (team_schedule.Week <= week_max) &
                                  ~(team_schedule.Team.isin(exclusions))]
    
    # Running All Combinations
    start = time.time()
    results = []
    results = weekly_pick(team_schedule=team_schedule,
                          week=team_schedule.Week.min(),
                          max_week=team_schedule.Week.max(),
                          picked_teams=[],
                          probs=[],
                          results=results,
                          start=start)
    if verbose:
        print('Running All Combs: %fs' %(time.time()-start))
    
    # Unpacking Dictionary
    start = time.time()
    picks = []
    probs_list = []
    probs = []
    for result in results:
        picks.append(result['picked_teams'])
        probs_list.append(result['probs'])
        probs.append(np.asarray(result['probs']).prod())
    if verbose:
        print('Unpacking Dict: %fs' %(time.time()-start))
    
    # Creating Data Frame
    start = time.time()
    results_table = pd.DataFrame()
    results_table['picks'] = picks
    results_table['probs'] = probs
    results_table['probs_list'] = probs_list
    results_table = results_table.sort_values(by='probs', ascending=False)
    if verbose:
        print('Creating Data Frame: %fs' %(time.time()-start))
    
    return results_table


def prediction_analysis(max_week=1):
    """
    function for analyzing prediction results
    """
    # Getting Data
    start = time.time()
    team_schedule = build_schedule(week='Week_1')
    for week in range(1, max_week):
        tmp_schedule = build_schedule(week='Week_%i'%week)
        team_schedule.loc[team_schedule.Week == week, 'win_pct'] = \
            tmp_schedule.loc[team_schedule.Week == week, 'win_pct']
    team_schedule = team_schedule[team_schedule.Week < max_week]
    print('Build Schedule: %fs' %(time.time()-start))
    
    team_schedule['win_pct_bin'] = team_schedule.win_pct.rank(pct=True)
    team_schedule.win_pct.hist(bins=np.arange(0.55,1,0.05), 
                               edgecolor='k', label='All Teams',
                               color='w')
    wins = team_schedule.win > 0
    team_schedule[wins].win_pct.hist(bins=np.arange(0.55,1,0.05), 
                                     edgecolor='k', label='Winning Teams',
                                     color='g')
    plt.xlabel('Win Probability')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    plt.show()
    
    return team_schedule


def optimization_analysis(max_weeks=2, week_adv=0):
    """
    run main for multiple weeks, check against schedule and see where it fails
    """
    team_schedule = build_schedule(week='Week_1')
    exclusions = []
    wins = []
    probs = []
    for week in range(1, max_weeks):
        result = main(week_min=week, week_max=week+week_adv, prob_min=0.65,
                            exclusions=exclusions, 
                            elo_week='Week_%i'%week)
        team = result.iloc[0, 0][0]
        wins.append(team_schedule[(team_schedule.Week == week)&
                                  (team_schedule.Team == team)].win.values[0])
        exclusions.append(team)
        probs.append(result.iloc[0, 2][0])
        # print(exclusions)
    df = pd.DataFrame()
    df['team'] = np.asarray(exclusions)
    df['win'] = np.asarray(wins)
    df['win_prob'] = np.asarray(probs)
    
    return df
        
for adv in range(7):
    results = optimization_analysis(max_weeks=12, week_adv=adv)
    print('\nAdvanced Look: %i weeks (%.3f)'
          %(adv, results.win_prob.product()))
    print(results)

team_schedule = prediction_analysis(max_week=11)

results = []
for i in range(11, 18):
    print('Week %i' %i)
    results.append(main(week_min=11, week_max=i, prob_min=0.65,
                        exclusions=['New Orleans', 'Dallas', 'Philadelphia',
                                    'Jacksonville', 'Carolina', 'Minnesota',
                                    'Atlanta', 'Pittsburgh', 'New England',
                                    'Kansas City'], 
                        elo_week='Week_11'), verbose=True)

for result in results:
    tmp = ''
    for i in range(len(result.iloc[0, 0])):
        tmp += result.iloc[0, 0][i] + ' (%.3f) '%result.iloc[0, 2][i]
    print(tmp+'\n')

for i in range(5):
    tmp = '%i. '%(i+1)
    for j in range(len(results[len(results)-1].iloc[0,0])):
        tmp += result.iloc[i, 0][j] + ' (%.3f) '%result.iloc[i, 2][j]
    print(tmp+'\n')
    

# Plots for later
"""
for i in range(10):
    plt.plot(np.arange(0, len(results[5].iloc[i,2])),
             np.asarray(results[5].iloc[i,2]),
             label='Pick #%i'%i)
plt.legend()
plt.show()
"""

# df10 = main(week_min=2, week_max=10, prob_min=0.70, 
#             exclusions=['New Orleans'], elo_week='Week_2')


# prob = 0
# iteration = 0
# for i in range(len(df.iloc[0,2])-1):
#     for j in range(i+1, len(df.iloc[0,2])):
#         prob += (1-df.iloc[0, 2][i])*(1-df.iloc[0, 2][j])
#         iteration += 1

# print(1-prob)

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
team_schedule[cond].elo_prob1.hist(
        bins=np.arange(0.55, 1, 0.025),
        edgecolor='k', label='All Teams', color='w')
team_schedule[cond&(team_schedule.result1 > 0.9)].elo_prob1.hist(
        bins=np.arange(0.55, 1, 0.025),
        edgecolor='k', label='Winning Teams', color='g')
plt.xlabel('Win Probability')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.show()

vals = []
bins = np.arange(0, .99, 0.025)
bin_labels = []
for i in range(len(bins)-1):
    bin_labels.append(bins[i])
    all_teams = team_schedule[(team_schedule.elo_prob1 >= bins[i])&
                              (team_schedule.elo_prob1 < bins[i+1])].shape[0]
    winning_teams = team_schedule[(team_schedule.elo_prob1 >= bins[i])&
                                  (team_schedule.elo_prob1 < bins[i+1])&
                                  (team_schedule.result1 > 0.9)].shape[0]
    if all_teams > 0:
        vals.append(winning_teams / all_teams)
    else:
        vals.append(0)

plt.plot(bins[:bins.shape[0]-1], vals)
plt.xlabel('ELO Win Probability')
plt.ylabel('Percentage of Actual Wins')
plt.show()

vals = []
bins = np.arange(0, .99, 0.025)
bin_labels = []
cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)
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
plt.ylabel('Percentage of Actual Wins')
plt.show()



cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)
for week in [1, 5, 10, 15]:
    vals = []
    bins = np.arange(0, .99, 0.1)
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(bins[i])
        all_teams = team_schedule[
                cond&
                (team_schedule.week == week)&
                (team_schedule.elo_prob1 >= bins[i])&
                (team_schedule.elo_prob1 < bins[i+1])
                ].shape[0]
        winning_teams = team_schedule[
                cond&
                (team_schedule.week == week)&
                (team_schedule.elo_prob1 >= bins[i])&
                (team_schedule.elo_prob1 < bins[i+1])&
                (team_schedule.result1 > 0.9)
                ].shape[0]
        if all_teams > 0:
            vals.append(winning_teams / all_teams)
        else:
            vals.append(0)
    
    plt.plot(bins[:bins.shape[0]-1], vals, label='Week %i'%week)
plt.xlabel('ELO Win Probability')
plt.ylabel('Percentage of Actual Wins')
plt.legend(loc='upper left')
plt.show()


cond = (team_schedule.season.isin(range(2007, 2018)))&\
       (team_schedule.playoff == 0)
for week in [1, 5, 10, 15]:
    vals = []
    bins = np.arange(0, .99, 0.1)
    bin_labels = []
    for i in range(len(bins)-1):
        bin_labels.append(bins[i])
        all_teams = team_schedule[
                cond&
                (team_schedule.week == week)&
                (team_schedule.elo_prob1 >= bins[i])&
                (team_schedule.elo_prob1 < bins[i+1])
                ].shape[0]
        winning_teams = team_schedule[
                cond&
                (team_schedule.week == week)&
                (team_schedule.elo_prob1 >= bins[i])&
                (team_schedule.elo_prob1 < bins[i+1])&
                (team_schedule.result1 > 0.9)
                ].shape[0]
        if all_teams > 0:
            vals.append(winning_teams / all_teams)
        else:
            vals.append(0)
    
    plt.plot(bins[:bins.shape[0]-1], vals, label='Week %i'%week)
plt.xlabel('ELO Win Probability')
plt.ylabel('Percentage of Actual Wins')
plt.legend(loc='upper left')
plt.show()

cond = (team_schedule.season == 2017)&\
       (team_schedule.playoff == 0)
team_sch = team_schedule[cond]
elo_week = pd.crosstab(index=team_sch.team1,
                       columns=team_sch.week,
                       values=team_sch.elo1,
                       aggfunc=np.nanmean)
elo_week['1']