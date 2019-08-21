import pandas as pd
from nfl_pickem import Pickem

if __name__ == "__main__":
    pk = Pickem()
    print(pk.file_path)
    df = pk.build_schedule()
    print(df[['team', 'win_pct', 'elo_prob1']])
