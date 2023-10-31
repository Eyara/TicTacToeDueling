import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from main import TicTacToeGame
from replay_manager import ReplayManager


class ExperienceReplayTrainer:
    def __init__(self):
        self.env = TicTacToeGame()
        self.device = "cpu"

        self.state = self.env.reset(True)
        self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.agent_x = DQNAgent(self.device, self.state, self.env.get_action_num(), "x")
        self.agent_o = DQNAgent(self.device, self.state, self.env.get_action_num(), "o")

    def learn_from_experience(self):
        replay_manager = ReplayManager()
        states, actions, rewards = replay_manager.get_all()
        for i in range(len(states)):
            for j in range(len(states[i])):
                state = torch.tensor(np.array(states[i][j][1]).flatten(), dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
                action = torch.tensor([[actions[i][j][1]]], device=self.device)
                next_state = torch.tensor([np.array([states[i][j + 1][1]]).flatten()], dtype=torch.float32,
                                          device=self.device) if j < len(states[i]) - 1 else None
                reward = torch.tensor([rewards[i][j][1]], device=self.device)
                if j % 2 == 0:
                    self.agent_x.learn(state, action, next_state, reward)
                else:
                    self.agent_o.learn(state, action, next_state, reward)

        torch.save(self.agent_x.policy_net.state_dict(), "./weights/pre_agent_x_policy.pt")
        torch.save(self.agent_x.target_net.state_dict(), "./weights/pre_agent_x_target.pt")
        torch.save(self.agent_o.policy_net.state_dict(), "./weights/pre_agent_o_policy.pt")
        torch.save(self.agent_o.target_net.state_dict(), "./weights/pre_agent_o_target.pt")


trainer = ExperienceReplayTrainer()
trainer.learn_from_experience()
