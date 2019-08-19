# NFL Pickem

Using FiveThrityEight's ELO ratings for NFL game predictions.  Historical data can be 
found 
[here](https://github.com/fivethirtyeight/nfl-elo-game/blob/master/data/nfl_games.csv).  
An explanation of the model can be 
[here](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).

## To-Do
- [ ] Analyze prediction accuracy for future games
- [ ] Build API structure
  - [ ] initialize by pulling data and creating table <br/>
  have file name/location be an attribute.
  - [ ] have method for calculating outcomes <br/>
  have attributes for prior picks, number of weeks to cover and
  and how many possible picks each week (or probability threshold)
  to consider
- [ ] Run analysis on historical outcomes for picking structure

## Analyzing ELO forecasts

Need to see how the actual vs. predicted probabilities change when considering
probabilities for future weeks (e.g. what's the performance when predicting for
the following week).  If the functional relationship remains linear, one can
apply a scaling coefficient to the model for predictions in advanced weeks.
The scaling coefficient will affect the overall probability of success for
a set of picks, potentially scaling down the need to save teams for future
weeks.

If, however, there is a non-linear relationship, then this optimization may
prove useful in taking advantage of such interaction.