import pulp as pl

import numpy as np
import pandas as pd

matches = [
    ['A', 'B', 1],
    ['C', 'D', 1],
    ['E', 'F', 1],
    ['C', 'A', 2],
    ['E', 'B', 2],
    ['F', 'D', 2],
    ['F', 'C', 3],
    ['E', 'A', 3],
    ['D', 'B', 3],
    ['A', 'D', 4],
    ['B', 'F', 4],
    ['C', 'E', 4],
    ['A', 'F', 5],
    ['B', 'C', 5],
    ['D', 'E', 5]
]

df = pd.DataFrame(matches)
df.columns = ['team1', 'team2', 'week']
df['win_pct'] = np.random.random(df.shape[0])
tmp = df.copy()
tmp.loc[:, 'win_pct'] = 1 - df.win_pct
df = pd.concat(
    [df,
     tmp.rename(columns={'team1': 'team2',
                        'team2': 'team1'})]
).reset_index()

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
prob.objective += pl.lpSum((np.log(df.win_pct.values[j]).round(4) * picks[j] for j in range(df.shape[0])))

prob.solve(pl.PULP_CBC_CMD(msg=1))

ind = []
for v in prob.variables():
    if v.varValue > 0:
        ind.append(int(v.name.split('_')[1]))

print(df.iloc[ind, :].sort_values(by='week'))
