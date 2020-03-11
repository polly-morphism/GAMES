import numpy as np
import pickle


class State:
    def __init__(self, agent, opponent):
        self.board = np.zeros((3, 3))
        self.agent = agent
        self.opponent = opponent
        self.endstate = False
        self.boardstate = None
        self.nowplaying = 1

    def boardhash(self):
        self.boardstate = str(self.board.reshape(3 * 3))
        return self.boardstate

    def available(self):
        positions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updatestate(self, position):
        self.board[position] = self.nowplaying
        self.boardstate = self.boardhash()
        self.nowplaying = -1 if self.nowplaying == 1 else 1

    def winner(self):
        for i in range(3):
            if sum(self.board[i, :]) == 3:
                self.endstate = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.endstate = True
                return -1
        for i in range(3):
            if sum(self.board[:, i]) == 3:
                self.endstate = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.endstate = True
                return -1
        diag_main = sum([self.board[i, i] for i in range(3)])
        diag_sub = sum([self.board[i, 3 - i - 1] for i in range(3)])
        diag_sum = max(abs(diag_main), abs(diag_sub))
        if diag_sum == 3:
            self.endstate = True
            if diag_main == 3 or diag_sub == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.available()) == 0:
            self.endstate = True
            return 0
        # not end
        self.endstate = False
        return None

    def playagent(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Epoch {}".format(i // 1000))
            while not self.endstate:
                positions = self.available()
                agent_action = self.agent.act(positions, self.board, self.nowplaying)
                self.updatestate(agent_action)
                board_hash = self.boardhash()
                self.agent.addstate(board_hash)

                win = self.winner()
                if win is not None:
                    self.rewards()
                    self.agent.reset()
                    self.opponent.reset()
                    self.reset()
                    break

                else:
                    positions = self.available()
                    opponent_action = self.opponent.act(
                        positions, self.board, self.nowplaying
                    )
                    self.updatestate(opponent_action)
                    board_hash = self.boardhash()
                    self.opponent.addstate(board_hash)

                    win = self.winner()
                    if win is not None:
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
            self.updatestate(agent_action)
            self.show()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.agent.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                positions = self.available()
                opponent_action = self.opponent.act(positions)

                self.updatestate(opponent_action)
                self.show()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.opponent.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def reset(self):
        self.board = np.zeros((3, 3))
        self.boardstate = None
        self.endstate = False
        self.nowplaying = 1

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

    def act(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
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
    # # training
    # agentx = Agent("agentx")
    # agento = Agent("agento")
    #
    # state = State(agentx, agento)
    # print("training start")
    # state.playagent(50000)
    # print("training end")

    agent = Agent("agent", exp=0)
    agent.loadpolicy("policy_agentx")

    human = Human("human")

    state = State(agent, human)
    game = 1
    while True:
        print("New game. Play #{}".format(game))
        state.playhuman()
        game += 1
