import random
from itertools import count

import matplotlib.pyplot as plt
import torch

from dqn_agent import DQNAgent, ReplayMemory
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

    # Take 100 episode averages and plot them too
    if len(reward_x) >= 50:
        means_x = reward_x.unfold(0, 50, 1).mean(1).view(-1)
        means_x = torch.cat((torch.zeros(49), means_x))
        plt.plot(means_x.numpy())

        means_o = reward_o.unfold(0, 50, 1).mean(1).view(-1)
        means_o = torch.cat((torch.zeros(49), means_o))
        plt.plot(means_o.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def get_next_state(done, observation):
    if done:
        return None
    else:
        return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def train_agent(agent: DQNAgent, current_state, action_sample, device):
    retry_count = 0
    while True:
        action = agent.select_action(current_state, action_sample)
        observation, reward, terminated, truncated = env.external_step(action.item())
        reward = torch.tensor([reward], device=device)
        current_done = terminated or truncated
        current_next_state = get_next_state(current_done, observation)
        agent_o.learn(current_state, action, current_next_state, reward)
        current_state = current_next_state

        # retry while the agent don't make a correct move
        if reward != -15 or retry_count > 5:
            break

        retry_count += 1

    return current_done, current_state, reward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_rewards_x = []
episode_rewards_o = []

if torch.cuda.is_available():
    num_episodes = 5000
else:
    num_episodes = 200

training_states = []

env = TicTacToeGame()
common_memory = ReplayMemory(1000000)

for i_episode in range(num_episodes):
    state = env.reset(True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    agent_x = DQNAgent(device, state, env.get_action_num())
    agent_o = DQNAgent(device, state, env.get_action_num())

    agent_x.memory.add_batch(common_memory.memory)
    agent_o.memory.add_batch(common_memory.memory)

    agents = [agent_x, agent_o]
    random.shuffle(agents)
    done = False

    for t in count():
        if not done:
            done, state, reward_x = train_agent(agents[0], state, env.get_action_sample(), device)
        else:
            reward_x = -10
        if not done:
            done, state, reward_o = train_agent(agents[1], state, env.get_action_sample(), device)
        else:
            reward_o = -10

        if done:
            episode_rewards_x.append(reward_x)
            episode_rewards_o.append(reward_o)
            plot_reward()
            break

    common_memory.add_batch(agent_x.memory.get_all())
    common_memory.add_batch(agent_o.memory.get_all())

# replay_manager = ReplayManager()
# replay_manager.save_to_file(training_states)

print('Complete')
plot_reward(show_result=True)
plt.ioff()
plt.show()
