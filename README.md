# NFL Pickem

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