import numpy as np
import pickle


class State:
    def __init__(self, agent, opponent):
        self.board = np.zeros((3, 3))
        self.agent = agent
        self.opponent = opponent
        self.round = 0
        self.endstate = False
        self.nowplaying = 1
        self.boardstate = None

    def boardhash(self):
        self.boardstate = str(self.board.reshape(3 * 3))
        return self.boardstate

    def available(self):
        positions = []
        for j in range(3):
            positions.append((self.round, j))
        return positions

    def updatestate(self, position, new_round):
        self.board[position] += self.nowplaying
        self.boardstate = self.boardhash()
        self.nowplaying = -1 if self.nowplaying == 1 else 1
        if new_round:
            self.round += 1

    def winner(self):
        if self.round == 2:
            self.endstate = True
            wins = 0
            for i in range(3):
                # Rock - Scissors
                if self.board[i, 0] == 1 and self.board[i, 2] == -1:
                    wins += 1
                elif self.board[i, 0] == -1 and self.board[i, 2] == 1:
                    wins -= 1
                # Paper - Rock
                if self.board[i, 1] == 1 and self.board[i, 0] == -1:
                    wins += 1
                elif self.board[i, 1] == -1 and self.board[i, 0] == 1:
                    wins -= 1
                # Scissors - Paper
                if self.board[i, 2] == 1 and self.board[i, 1] == -1:
                    wins += 1
                elif self.board[i, 2] == -1 and self.board[i, 2] == 1:
                    wins -= 1
            # 1 player wins
            if wins > 0:
                return 1
            # 2 player wins
            elif wins < 0:
                return -1
            # tie
            elif wins == 0:
                return 0
        # not end
        self.endstate = False
        return None

    def playagent(self, rounds=100):
        for i in range(rounds):

            while not self.endstate:
                positions = self.available()
                agent_action = self.agent.act(positions, self.board, self.nowplaying)
                self.updatestate(agent_action, False)

                positions = self.available()
                opponent_action = self.opponent.act(positions, self.round)
                self.updatestate(opponent_action, True)
                board_hash = self.boardhash()
                self.opponent.addstate(board_hash)
                self.agent.addstate(board_hash)
                self.show()
                win = self.winner()
                if win is not None:
                    if win == 1:
                        print(self.agent.name, "wins!")
                    elif win == -1:
                        print(self.opponent.name, "wins!")
                    else:
                        print("tie!")
                    self.rewards()
                    self.agent.reset()
                    self.opponent.reset()
                    self.reset()
                    break

        self.agent.savepolicy()

    def playhuman(self):
        while not self.endstate:
            positions = self.available()
            agent_action = self.agent.act(positions, self.board, self.nowplaying)
            self.updatestate(agent_action, False)
            positions = self.available()
            opponent_action = self.opponent.act(positions, self.round)

            self.updatestate(opponent_action, True)
            self.show()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.agent.name, "wins!")
                elif win == -1:
                    print(self.opponent.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

    def reset(self):
        self.board = np.zeros((3, 3))
        self.boardstate = None
        self.endstate = False
        self.round = 0

    def rewards(self):
        result = self.winner()
        if result == 1:
            self.agent.reward(1)
            self.opponent.reward(0)
        elif result == -1:
            self.agent.reward(0)
            self.opponent.reward(1)
        else:
            self.agent.reward(0.1)
            self.opponent.reward(0.5)

    def show(self):
        for i in range(0, 3):
            print("R-----P-----S")
            print("-------------")
            out = "| "
            for j in range(0, 3):
                if self.board[i, j] == 1:
                    token = "x"
                if self.board[i, j] == -1:
                    token = "o"
                if self.board[i, j] == 0:
                    token = " "
                out += token + " | "
            print(out)
        print("-------------")


class Agent:
    def __init__(self, name, exp=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp = exp
        self.decaygamma = 0.9
        self.statesvalue = {}

    def boardstate(self, board):
        boardstate = str(board.reshape(3 * 3))
        return boardstate

    def act(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardstate = self.boardstate(next_board)
                value = (
                    0
                    if self.statesvalue.get(next_boardstate) is None
                    else self.statesvalue.get(next_boardstate)
                )
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def reward(self, reward):
        for state in reversed(self.states):
            if self.statesvalue.get(state) is None:
                self.statesvalue[state] = 0
            self.statesvalue[state] += self.lr * (
                self.decaygamma * reward - self.statesvalue[state]
            )
            reward = self.statesvalue[state]

    def addstate(self, state):
        self.states.append(state)

    def reset(self):
        self.states = []

    def savepolicy(self):
        fw = open("policy_" + str(self.name), "wb")
        pickle.dump(self.statesvalue, fw)
        fw.close()

    def loadpolicy(self, file):
        fr = open(file, "rb")
        self.statesvalue = pickle.load(fr)
        fr.close()


class Human:
    def __init__(self, name):
        self.name = name

    def act(self, positions, round):
        while True:
            action = input("Input your action:")
            if action == "R":
                row = round
                col = 0
            elif action == "P":
                row = round
                col = 1
            elif action == "S":
                row = round
                col = 2
            action = (row, col)
            if action in positions:
                return action

    def addstate(self, state):
        pass

    def reward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    # training
    agent = Agent("agent")
    human = Human("human")

    state = State(agent, human)
    print("training start")
    state.playagent(30)
    print("training end")

    agent = Agent("agent", exp=0)
    agent.loadpolicy("policy_agent")

    human = Human("human")

    state = State(agent, human)
    game = 1
    while True:
        print("New game. Play #{}".format(game))
        state.playhuman()
        game += 1
