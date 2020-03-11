
Let's define the Markov Decision Process for this game.



STATE

The state is a vector of 6 numbers representing last 3 rounds, one round consists of 2 players simultaneously playing rock, paper or scissors.
State variables are: rounds (first it's a vector of zeros and then it changes), a variable representing if the game over or not.

Assume each play takes 3 rounds and in the end we define the winner by whoever has 2 or more wins, so, player A wins if he has 2 points, and player B
has 1/2 / player A has 1 and player B has 0/ player A has 3.
All positions with 0 in them are available, all positions with 1 have X and with -1 have O.
We reset the board to it's starting state in the end of the game.

ACTION



PROBABILITY FUNCTION



REWARDS



VALUE

the updated value of state t equals the current value of state t adding the difference between the value of next state and the value of current state, which is multiplied by a learning rate α
(Given the reward of intermediate state is 0).
