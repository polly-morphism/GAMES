import gym
import numpy as np
import torch
from datetime import datetime
from gym import wrappers
from time import time


def timeit(func):
    def timeit_wr(*args, **kwargs):
        st = time()
        func(*args, **kwargs)
        print(f"dt={time()-st}")

    return timeit_wr


# @timeit
def preprocess_states(input_state, prev_processed_state, input_dimensions):
    """
    convert the 210x160x3 uint8 frame into a 6400 float vector
    """
    processed_state = input_state[35:195]
    processed_state = processed_state[::2, ::2, :]
    processed_state = processed_state[:, :, 0]
    processed_state[processed_state == 144] = 0
    processed_state[processed_state == 109] = 0
    processed_state[processed_state != 0] = 1
    # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_state = processed_state.astype(np.float).ravel()

    # subtract the previous frame from the current one
    # so we are only processing on changes in the game
    if prev_processed_state is not None:
        input_state = processed_state - prev_processed_state
    else:
        input_state = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_states = processed_state
    return input_state, prev_processed_states


def relu(vector):
    vector[vector < 0] = 0
    return vector


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# @timeit
def apply_neural_nets(state_matrix, weights, device):
    """
    Based on the state_matrix and weights,
    compute the new hidden layer values and the new output layer values
    """
    if device == "cpu":
        # Compute the unprocessed hidden layer values by simply finding the dot
        # product of the weights (weights of layer 1) and the state_matrix.
        hidden_layer_values = np.dot(weights["1"], state_matrix)
        # apply a non linear thresholding function on those hidden layer values
        # in this case just a simple ReLU.
        # This introduces the nonlinearities that makes our network
        # capable of computing nonlinear functions
        hidden_layer_values = relu(hidden_layer_values)
        # We use those hidden layer activation values
        # to calculate the output layer values.
        # This is done by a simple dot product of hidden_layer_values (200 x 1)
        # and weights['2'] (1 x 200) which yields a single value (1 x 1).
        output_layer_values = np.dot(hidden_layer_values, weights["2"])
        # apply a sigmoid function on this output value so that
        # it's between 0 and 1 and is therefore a valid probability of going up
        output_layer_values = sigmoid(output_layer_values)
        return hidden_layer_values, output_layer_values
    elif device == "cuda":
        # Compute the unprocessed hidden layer values by simply finding the dot
        # product of the weights (weights of layer 1) and the state_matrix.
        hidden_layer_values = torch.matmul(weights["1"], state_matrix)
        # apply a non linear thresholding function on those hidden layer values
        # in this case just a simple ReLU.
        # This introduces the nonlinearities that makes our network
        # capable of computing nonlinear functions
        hidden_layer_values = relu(hidden_layer_values)
        # We use those hidden layer activation values
        # to calculate the output layer values.
        # This is done by a simple dot product of hidden_layer_values (200 x 1)
        # and weights['2'] (1 x 200) which yields a single value (1 x 1).
        output_layer_values = torch.matmul(hidden_layer_values, weights["2"])
        # apply a sigmoid function on this output value so that
        # it's between 0 and 1 and is therefore a valid probability of going up
        output_layer_values = sigmoid(output_layer_values.cpu().numpy())
        return hidden_layer_values, output_layer_values


# @timeit
def compute_gradient(
    gradient_log_p, hidden_layer_values, state_values, weights, device
):
    if device == "cpu":
        delta_L = gradient_log_p
        dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
        delta_l2 = np.outer(delta_L, weights["2"])
        delta_l2 = relu(delta_l2)
        dC_dw1 = np.dot(delta_l2.T, state_values)
        return {"1": dC_dw1, "2": dC_dw2}
    elif device == "cuda":
        delta_L = gradient_log_p
        dC_dw2 = torch.matmul(hidden_layer_values.T, delta_L)
        delta_l2 = torch.ger(delta_L, weights["2"])
        delta_l2 = relu(delta_l2)
        dC_dw1 = torch.matmul(delta_l2.T, state_values)
        return {"1": dC_dw1, "2": dC_dw2}


# @timeit
def update_weights(
    weights, expectation_g_squared, g_dict, decay_rate, learning_rate, device
):
    epsilon = 1e-5
    if str(device) == "cpu":
        for layer_name in weights.keys():
            g = g_dict[layer_name]
            expectation_g_squared[layer_name] = (
                decay_rate * expectation_g_squared[layer_name]
                + (1 - decay_rate) * g ** 2
            )
            weights[layer_name] += (learning_rate * g) / (
                np.sqrt(expectation_g_squared[layer_name] + epsilon)
            )
            g_dict[layer_name] = np.zeros_like(
                weights[layer_name]
            )  # reset batch gradient buffer
    elif str(device) == "cuda":
        for layer_name in weights.keys():
            g = g_dict[layer_name]
            expectation_g_squared[layer_name] = (
                decay_rate * expectation_g_squared[layer_name]
                + (1 - decay_rate) * g ** 2
            )
            weights[layer_name] += (learning_rate * g) / (
                torch.sqrt(expectation_g_squared[layer_name] + epsilon)
            )
            g_dict[layer_name] = torch.zeros_like(
                weights[layer_name], device=device, dtype=torch.float
            )  # reset batch gradient buffer


# @timeit
def discount_rewards(rewards, gamma, device):
    """
    Actions you took 20 steps before the end result are less important to
    the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions
    based on how long ago they were taken
    """
    if str(device) == "cpu":
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    elif str(device) == "cuda":
        discounted_rewards = torch.zeros_like(rewards, device=device, dtype=torch.float)
        running_add = 0
        for t in reversed(range(0, rewards.size()[0])):
            if rewards[t] != 0:
                # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


# @timeit
def discount_with_rewards(gradient_log_p, episode_rewards, gamma, device):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma, device)

    if str(device) == "cpu":
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return gradient_log_p * discounted_episode_rewards
    elif str(device) == "cuda":
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_rewards -= torch.mean(discounted_episode_rewards)
        discounted_episode_rewards /= torch.std(discounted_episode_rewards)
        return gradient_log_p * discounted_episode_rewards


def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3


def train_cpu(n_epoch):
    n_epoch = n_epoch
    env = gym.make("Pong-v0")
    state = env.reset()  # state
    batch_size = 10  # how many episodes to wait before moving the weights
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200  # number of neurons
    input_dimensions = 80 * 80  # dimension of our state images
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_states = None
    device = torch.device("cpu")
    weights = {
        "1": np.random.randn(num_hidden_layer_neurons, input_dimensions)
        / np.sqrt(input_dimensions),
        "2": np.random.randn(num_hidden_layer_neurons)
        / np.sqrt(num_hidden_layer_neurons),
    }

    # rmsprop
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    (
        episode_hidden_layer_values,
        episode_states,
        episode_gradient_log_ps,
        episode_rewards,
    ) = ([], [], [], [])
    epoch = 0
    while epoch <= n_epoch:
        env.render()
        processed_states, prev_processed_states = preprocess_states(
            state, prev_processed_states, input_dimensions
        )
        hidden_layer_values, up_probability = apply_neural_nets(
            processed_states, weights, "cpu"
        )

        episode_states.append(processed_states)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        state, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        if done:  # an episode finished
            episode_number += 1
            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_states = np.vstack(episode_states)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(
                episode_gradient_log_ps, episode_rewards, gamma, device
            )

            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_states,
                weights,
                "cpu",
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                print("New Batch")
                update_weights(
                    weights,
                    expectation_g_squared,
                    g_dict,
                    decay_rate,
                    learning_rate,
                    device,
                )

            (
                episode_hidden_layer_values,
                episode_states,
                episode_gradient_log_ps,
                episode_rewards,
            ) = (
                [],
                [],
                [],
                [],
            )  # reset values
            state = env.reset()  # reset env
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )
            print(
                "CPU resetting env. episode reward total was {}. running mean: {}".format(
                    reward_sum, running_reward,
                )
            )
            reward_sum = 0
            prev_processed_states = None
            epoch -= -1


def train_gpu(n_epoch):
    n_epoch = n_epoch
    env = gym.make("Pong-v0")
    state = env.reset()  # state
    batch_size = 10  # how many episodes to wait before moving the weights
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200  # number of neurons
    input_dimensions = 80 * 80  # dimension of our state images
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_states = None
    device = torch.device("cuda")
    weights = {
        "1": torch.randn(
            num_hidden_layer_neurons,
            input_dimensions,
            device=device,
            dtype=torch.float,
        )
        / np.sqrt(input_dimensions),
        "2": torch.randn(num_hidden_layer_neurons, device=device, dtype=torch.float)
        / np.sqrt(num_hidden_layer_neurons),
    }

    # rmsprop
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = torch.zeros_like(
            weights[layer_name], device=device, dtype=torch.float
        )
        g_dict[layer_name] = torch.zeros_like(
            weights[layer_name], device=device, dtype=torch.float
        )

    (
        episode_hidden_layer_values,
        episode_states,
        episode_gradient_log_ps,
        episode_rewards,
    ) = ([], [], [], [])
    epoch = 0
    while epoch <= n_epoch:
        env.render()
        processed_states, prev_processed_states = preprocess_states(
            state, prev_processed_states, input_dimensions
        )
        processed_states = torch.from_numpy(processed_states).float().to(device)
        hidden_layer_values, up_probability = apply_neural_nets(
            processed_states, weights, "cuda"
        )
        episode_states.append(processed_states)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        state, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        if done:  # an episode finished
            episode_number += 1
            episode_gradient_log_ps = torch.FloatTensor(episode_gradient_log_ps).to(
                device
            )
            episode_rewards = torch.FloatTensor(episode_rewards).to(device)
            # Combine the following values for the episode
            episode_hidden_layer_values = torch.stack(episode_hidden_layer_values)
            episode_states = torch.stack(episode_states)
            # episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            # episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(
                episode_gradient_log_ps, episode_rewards, gamma, device
            )

            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_states,
                weights,
                "cuda",
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                print("New Batch")
                update_weights(
                    weights,
                    expectation_g_squared,
                    g_dict,
                    decay_rate,
                    learning_rate,
                    device,
                )

            (
                episode_hidden_layer_values,
                episode_states,
                episode_gradient_log_ps,
                episode_rewards,
            ) = (
                [],
                [],
                [],
                [],
            )  # reset values
            state = env.reset()  # reset env
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )
            print(
                "GPU resetting env. episode reward total was {}. running mean: {}".format(
                    reward_sum, running_reward,
                )
            )
            reward_sum = 0
            prev_processed_states = None
            epoch -= -1


if __name__ == "__main__":
    start = datetime.now()
    train_cpu(2)
    end = datetime.now()
    print("CPU", end - start)

    start = datetime.now()
    train_gpu(2)
    end = datetime.now()
    print("GPU", end - start)
