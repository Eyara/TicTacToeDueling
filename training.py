import random
from itertools import count

import matplotlib.pyplot as plt
import torch

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from main import TicTacToeGame
from agents.minimax_agent import MiniMaxAgent
from replay_manager import ReplayManager


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
        plt.plot(means_x.numpy(), label="Means X")

        means_o = reward_o.unfold(0, 50, 1).mean(1).view(-1)
        means_o = torch.cat((torch.zeros(49), means_o))
        plt.plot(means_o.numpy(), label="Means O")
        plt.legend(loc='best')

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_stats():
    plt.figure(1)
    wins_x_stat = torch.tensor(wins_x, dtype=torch.float)
    wins_o_stat = torch.tensor(wins_o, dtype=torch.float)
    draws_stat = torch.tensor(draws, dtype=torch.float)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.plot(wins_x_stat.numpy(), label="Wins X")
    plt.plot(wins_o_stat.numpy(), label="Wins O")
    plt.plot(draws_stat.numpy(), label="Draws")

    plt.legend(loc='best')
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_next_state(done, observation):
    if done:
        return None
    else:
        return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def train_agent(agent, current_state, action_sample, device, is_minimax=False):
    retry_count = 0
    while True:
        if is_minimax:
            action = agent.select_action(env, action_sample)
        else:
            action = agent.select_action(current_state, action_sample)
        observation, reward, terminated, truncated = env.external_step(action.item())

        # smells bad but does not care tbh
        training_states.append([i_episode, observation.tolist(), action.item(), reward])

        reward = torch.tensor([reward], device=device)
        current_done = terminated or truncated
        current_next_state = get_next_state(current_done, observation)
        agent.learn(current_state, action, current_next_state, reward)
        current_state = current_next_state

        # retry while the agent don't make a correct move
        if reward != -15 or retry_count > 5:
            break

        retry_count += 1

    return current_done, current_state, reward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_rewards_x = []
episode_rewards_o = []

wins_x_count = 0
wins_o_count = 0
draws_count = 0

wins_x = []
wins_o = []
draws = []

if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 200

training_states = []

env = TicTacToeGame()

state = env.reset(True)
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
agent_x = DQNAgent(device, state, env.get_action_num(), "x")
# agent_o = DQNAgent(device, state, env.get_action_num(), "o")
# agent_x = MiniMaxAgent(env.get_action_num(), 1)
agent_o_1 = MiniMaxAgent(env.get_action_num(), 2)
# agent_x = RandomAgent(env.get_action_num())
agent_o_2 = RandomAgent(env.get_action_num())

for i_episode in range(num_episodes + 1):
    state = env.reset(True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    agent_bots = [agent_o_1, agent_o_2]
    random.shuffle(agent_bots)

    agents = [agent_x, agent_bots[0]]
    # random.shuffle(agents)
    done = False

    for t in count():
        if not done:
            done, state, reward_x = train_agent(agents[0], state, env.get_action_sample(), device)
        if not done:
            done, state, reward_o = train_agent(agents[1], state, env.get_action_sample(), device, True)

        if done:
            if reward_x == 10:
                wins_x_count += 1
            elif reward_o == 10:
                wins_o_count += 1
            else:
                draws_count += 1

            wins_x.append(wins_x_count)
            wins_o.append(wins_o_count)
            draws.append(draws_count)

            episode_rewards_x.append(reward_x)
            episode_rewards_o.append(reward_o)
            print(wins_x_count, wins_o_count, draws_count)
            # plot_stats()
            break

    if i_episode % 100 == 0:
        torch.save(agent_x.policy_net.state_dict(), "./weights/agent_x_policy_4.pt")
        torch.save(agent_x.target_net.state_dict(), "./weights/agent_x_target_4.pt")
    #     torch.save(agent_o.policy_net.state_dict(), "./weights/agent_o_policy.pt")
    #     torch.save(agent_o.target_net.state_dict(), "./weights/agent_o_target.pt")

replay_manager = ReplayManager()
replay_manager.save_to_file(training_states)

# plot_reward(show_result=True)
plt.ioff()
plt.show()
