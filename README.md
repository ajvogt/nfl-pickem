# NFL Pickem
Author: Adam J. Vogt

The goal of this project is to make an optimal selection of 
sequential picks of which NFL team will win their particular 
matchup each week.  

A [quickstart](#quickstart) guide can be found here.

## Context: Survivor Pool
In a survivor pool game, each player must 
select one NFL team each week that the player thinks 
will win their particular NFL matchup.
Multiple players may select the same 
team in a particular week.
Players are eliminated from the pool when they reach 
a set threshold of incorrect picks (e.g. Player A is eliminated 
in week 6 after 
making two incorrect picks, one in week 4 and the other in 
week 6).
The other constraint is that each Player can maximally pick each 
team once throughout the season (e.g. Player B selected the 
New England Patriots last week and therefore cannot select them 
in future weeks.)

Given these constraints, a Player would like to make a sequence 
of picks where their selected NFL teams will have the lowest 
liklihood of losing (or in some cases drawing) in the week of 
selection.  Because each team can only be selected once, there 
is a tradeoff between selecting the team with highest liklihood 
of winning this week or saving that team for a future week in 
which there are worse alternative matchups.

The final consideration is the estimation of a team's liklihood 
of success in this week versus in that of a future week.  Both 
the magnitude of the likelihood and the confidence in that 
estimation for future weeks are factors that affect this 
selection tradeoff.

## Methods
Two things are needed to make an optimal selection of 
sequential picks:
1. a numerical estimation of the liklihood that a team will 
win thier particular matchup in a given week, and 
2. an objective to maximize that will increase chances of 
survival in the pool.

For the first point, we use FiveThrityEight's ELO ratings model 
for NFL game predictions. 
The ELO ratings are combined with home field 
status to estimate the liklihood that a team will win their 
particular matchup.  Since the rating does not use time or 
week of the season as an input, the current rating can be used 
to estimate that probability for any future matchup.  Ratings 
are adjusted after each week to account for the team's latest 
performance. 
Historical data can be 
found 
[here](https://github.com/fivethirtyeight/nfl-elo-game/blob/master/data/nfl_games.csv). 
An explanation of the model can be 
[here](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).

For the second point, we then use these win probabilities 
to build an objective.  We use a linear mixed-integer 
optimization to maximize the log of the product of the win 
probabilities for the sequence of picks. 
This approximates the probability that all picks will be 
a success and that there will be no incorrect selections. 
The problem is 
constrained by the bounds of the survivor pool game; 
1. One and only one team/matchup can be selected each week, and 
2. Each team can be selected maximally one time.

Because of the uncertainty in the estimation of future 
liklihood of success, optimal sequences of any length can 
be calculated and the risks of trading present liklihoods 
versus future ones can be left to the player.

## To-Do
- [x] Build API structure
  - [x] initialize by pulling data and creating table <br/>
  have file name/location be an attribute.
  - [x] have method for calculating outcomes <br/>
  have attributes for prior picks, number of weeks to cover and
  and how many possible picks each week (or probability threshold)
  to consider
- [x] create script to compare forecasts of picks
- [ ] Run analysis on historical outcomes for picking structure
- [ ] Analyze prediction accuracy for future games

<a name="quickstart"></a>

## Quick Start Guide



## Discussion & Ideas

### Notes on Future Predictability
By considering all week10+ regular season games from 2007 to 2018, if I estimate a
team's probability of winning a match based on their current elo rating vs. ratings from
weeks 1-7 prior, the log-loss of the prediction model is pretty stable.  That said, 
the average probability change for all teams between the current rating and a previous 
one is on the order of 0.1%.  There is either some stability in a lot of the ELO ratings 
or the zero-sum nature of the rating system causes fluctuations up and down to balance. 
That said, the RMSE was still on the order of 1%.

### Analyzing ELO forecasts

Need to see how the actual vs. predicted probabilities change when considering
probabilities for future weeks (e.g. what's the performance when predicting for
the following week).  If the functional relationship remains linear, one can
apply a scaling coefficient to the model for predictions in advanced weeks.
The scaling coefficient will affect the overall probability of success for
a set of picks, potentially scaling down the need to save teams for future
weeks.

If, however, there is a non-linear relationship, then this optimization may
prove useful in taking advantage of such interaction.