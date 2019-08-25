"""python script used for development
"""

import pandas as pd
from nfl_pickem import Pickem

if __name__ == "__main__":
    pk = Pickem()
    print(pk.file_path)
    pk.pull_data()
    print('Current Teams')
    df = pk.build_schedule(season=2017)
    teams = np.unique(np.concatenate(
        (df.team1, df.team2)
    ))
    print(teams)
    pk.compare_picks(season=2017,
                     current_week=9,
                     prior_picks=['NE', 'NO', 'SEA', 'BAL', 'LAR',
                                  'DEN', 'JAX', 'PIT'])
