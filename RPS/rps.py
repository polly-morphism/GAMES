import random
import operator
import itertools

# INITIALIZE RULES #

moves = ["R", "P", "S"]
wins = {"S": "R", "R": "P", "P": "S"}
rewards = {
    "RR": 0,
    "PP": 0,
    "SS": 0,
    "PR": 1,
    "SP": 1,
    "RS": 1,
    "RP": -1,
    "PS": -1,
    "SR": -1,
}

max_game_len = 100

game_chain = []
opponents_chain = []

weights_for_descisions = {
    "RR": {"R": 0, "P": 0, "S": 0},
    "PP": {"R": 0, "P": 0, "S": 0},
    "SS": {"R": 0, "P": 0, "S": 0},
    "PR": {"R": 0, "P": 0, "S": 0},
    "SP": {"R": 0, "P": 0, "S": 0},
    "RS": {"R": 0, "P": 0, "S": 0},
    "RP": {"R": 0, "P": 0, "S": 0},
    "PS": {"R": 0, "P": 0, "S": 0},
    "SR": {"R": 0, "P": 0, "S": 0},
}

####################


def get_winning(move):
    return wins[move]


def get_most_common_patterns(opponents_chain=opponents_chain, length=3):
    chain = "".join(opponents_chain)
    subchains = {}
    for i in range(0, len(chain) - length):
        subchain = chain[i : i + length]
        if subchain not in subchains:
            subchains[subchain] = 1
        else:
            subchains[subchain] += 1
    subchains = {
        k: v for k, v in sorted(subchains.items(), key=lambda item: item[1]) if v > 1
    }
    return dict(itertools.islice(subchains.items(), 5))


def make_a_choice(game_chain=game_chain, moves=moves):
    if len(game_chain) >= 9:
        opponents_strategies = get_most_common_patterns()
        print("STRATEGIES", opponents_strategies)
        if len(opponents_strategies) >= 1:
            last_moves = "".join(opponents_chain[-2:])
            for key, value in opponents_strategies.items():
                if key[:2] == last_moves:
                    print("WINNING", get_winning(key[-1]))
                    return get_winning(key[-1])
            return random.choice(moves)
        else:
            return random.choice(moves)

    else:
        return random.choice(moves)


# def make_a_choice(game_chain=game_chain, moves=moves):
#     if len(game_chain) >= 1:
#         last_choice = game_chain[-1]
#         my_choice = max(
#             weights_for_descisions[last_choice].items(), key=operator.itemgetter(1)
#         )[0]
#         if weights_for_descisions[last_choice][my_choice] <= 0:
#             my_choice = random.choice(moves)
#     else:
#         my_choice = random.choice(moves)
#     return my_choice
#
#
def game(rewards=rewards, max_game_len=max_game_len):

    my_choice = make_a_choice()
    opponents_choice = input("Rock, Paper, Scissors ...  Your choice: ")

    print(
        "Agents reward: {}".format(rewards[my_choice + opponents_choice]),
        "Agents choice: {}; Your choice: {}".format(my_choice, opponents_choice),
    )
    # print("\n\n", weights_for_descisions)

    # weights_for_descisions[my_choice + opponents_choice][my_choice] += rewards[
    #     my_choice + opponents_choice
    # ]

    game_chain.append(my_choice + opponents_choice)
    opponents_chain.append(opponents_choice)
    print("Moves so far: ", game_chain[-5:])


game_len = 0

while game_len <= max_game_len:
    game()
    # if game_len == max_game_len:
    #     print(weights_for_descisions)
    game_len += 1
