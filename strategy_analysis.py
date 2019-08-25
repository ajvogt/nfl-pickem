from multiprocessing import Pool
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

from nfl_pickem import Pickem

def run_strategy(pk, max_week, season):
    picks = pd.DataFrame()
    columns = ['team1', 'win_prob', 'result1', 'week']
    for week in range(1, 18):
        ts = pk.build_schedule(season=season,
                               elo_week=week)
        if week > 1:
            ts = ts[~ts.team1.isin(picks.team1)].reset_index()
        max_week = min(ts.week.max(), max_week)
        ind = pk.pick_optimization(
            ts[ts.week <= week + max_week]
        )
        picks = picks.append(ts.loc[ind[0], columns])
    
    return picks


def run_season(season):
    pk = Pickem()
    pk.pull_data()
    
    start = time.time()
    
    tmp = pk.build_schedule(season=season)
    reg_season_len = tmp[tmp.playoff < 0.9].week.max()
    results = []
    for max_week in range(0, reg_season_len):
        ind_start = time.time()
        picks = run_strategy(pk,
                             max_week=max_week,
                             season=season)
        status = '%i, '%season
        status += 'Max Week: %i, '%max_week
        status += 'Time: %.3fs, '%(time.time()-ind_start)
        status += 'Correct: %i/%i, '%(picks.result1.sum(), 
                                      picks.shape[0])
        if picks.result1.sum() < 16:
            elim_week = picks[picks.result1 < 0.9].week.values[1]
            status += 'Elimination Week: %i'%(elim_week)
        else:
            elim_week = np.nan

        results.append([season, max_week, picks.result1.sum(),
                        picks.shape[0], elim_week])

        print(status)
    print('Total time: %.3fs'%(time.time()-start))

    return results


if __name__ == '__main__':
    p = Pool(4)
    results = p.map(run_season, range(1997, 2018))
    for i in range(len(results)):
        results[i] = pd.DataFrame(results[i])
        results[i].columns = ['season', 'max_week', 'correct', 
                              'possible', 'elim_week']
    
    df = pd.concat([i for i in results])
    df.to_csv('results/strategy_analysis.csv', index=False)
    print('Final Records: %i'%df.shape[0])
