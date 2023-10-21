from itertools import count

import matplotlib.pyplot as plt
import torch

from dqn_agent import DQNAgent
from main import TicTacToeGame


# from replay_manager import ReplayManager


def plot_reward(show_result=False):
    plt.figure(1)
    reward_x = torch.tensor(episode_rewards_x, dtype=torch.float)
    reward_o = torch.tensor(episode_rewards_o, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_x.numpy())
    plt.plot(reward_o.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_x) >= 100:
        means = reward_x.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def get_next_state(done, observation):
    if done:
        return None
    else:
        return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_rewards_x = []
episode_rewards_o = []

if torch.cuda.is_available():
    num_episodes = 3000
else:
    num_episodes = 50

training_states = []

env = TicTacToeGame()

for i_episode in range(num_episodes):
    state = env.reset(True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    agent_x = DQNAgent(device, state, env.get_action_num())
    agent_o = DQNAgent(device, state, env.get_action_num())
    for t in count():
        action_x = agent_x.select_action(state, env.get_action_sample())

        observation_x, reward_x, terminated_x, truncated_x = env.external_step(action_x.item())

        reward_x = torch.tensor([reward_x], device=device)
        done = terminated_x or truncated_x

        # training_states.append((i_episode, observation))

        next_state = get_next_state(done, observation_x)
        agent_x.learn(state, action_x, next_state, reward_x)
        state = next_state

        # training the second agent if first don't win on this turn
        if not done:
            action_o = agent_o.select_action(state, env.get_action_sample())
            observation_o, reward_o, terminated_o, truncated_o = env.external_step(action_o.item())
            reward_o = torch.tensor([reward_o], device=device)
            done = terminated_o or truncated_o
            next_state = get_next_state(done, observation_o)
            agent_o.learn(state, action_o, next_state, reward_o)
        else:
            reward_o = torch.tensor([-10], device=device)

        state = next_state

        if done:
            episode_rewards_x.append(reward_x)
            episode_rewards_o.append(reward_o)
            plot_reward()
            break

# replay_manager = ReplayManager()
# replay_manager.save_to_file(training_states)

print('Complete')
plot_reward(show_result=True)
plt.ioff()
plt.show()
