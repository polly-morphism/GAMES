The game of Tic-Tac-Toe is a perfect example of adversarial non-cooperative zero-sum game. The brute-force way to solve ttt is provided by combinatorial Game Theory –
 a branch of mathematics that analyses all different outcomes of an event. It can be implemented easily in this form, but iteration through all possible X and O positions is
obviously not the most elegant way of solving matrix games.

Tic-Tac-Toe combinatorics are:
3^9= 19,683  -  total number of possible game positions in a 3×3 grid – as every square will either be a O, X or blank.

9! = 362,880 - total number of ways that positions can be filled on the grid. (First you have 9 choices of squares, then there are 8 choices of squares etc).  This counts each X and O as distinct from other X and Os.

9 choose 5 = 126 - number of different combinations of filling the grid with 5 Xs and 4 Os.  

So, lets look at ttt from reinforcement learning perspective.

Our agent is playing with an opponent. We do not make any assumptions about opponents strategy (as opposed to classical game theory that assumes a pro-opponent), we are just trying to maximize our own
rewards.

The mathematical complexity is also reduced by avoiding any model of environment. Out opponent is the environment, and everything else just doesn't matter to us.

Let's first develop two policies with multi-agent Q-learning and then move on to play against humans.

Let's define the Markov Decision Process for this game.
We play with X and the second player plays with O (if it's a game against a human, not two agents). X's have the advantage of making the first move, but, whatever)

STATE

The state is a vector of 9 numbers (3x3 matrix) representing the positions Os Xs and Os in the moment t.
State variables are: board (first it's a matrix of zeros and then it changes), the player that has to make a move now (starts with player x). We also define the variable
that is True if the game is over and 0 if not.

We determine the winner or tie by getting sums on rows, columns and diagonal sums. So, if the X is in position 1,1; 1,2; 1,3 - we have +1 in them and the sum id 3. So X is a winner, etc.
All positions with 0 in them are available, all positions with 1 have X and with -1 have O.
We reset the board to it's starting state in the end of the game.

ACTION

Action space of player X is a set of all empty positions if it's X's turn and the game does not have a winner.
The play goes like this: get state, get available actions, choose actions, collect rewards, continue until the winner is determined.

PROBABILITY FUNCTION

X takes an action, the environment responds and returns a new state after O’s turn. Every possible state has its own probability and the probability depends on O’s player.
O's actions change the state and therefore the action-set for X, hence O-player influences probabilities.

REWARDS

If the agent wins the game, it is given a reward of 1 and the other player gets 0 and visa versa. Is it's a tie - the agent gets 0.1 and the other player gets 0.5 (we think of
ties as not much better then loosing).
Rewards are given in the end of the game (rewards are hyperparameters in our case and can be tuned).

VALUE

the updated value of state t equals the current value of state t adding the difference between the value of next state and the value of current state, which is multiplied by a learning rate α
(Given the reward of intermediate state is 0).
