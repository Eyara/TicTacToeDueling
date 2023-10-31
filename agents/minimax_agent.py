import random
from copy import deepcopy

import torch


class MiniMaxAgent:
    def __init__(self, action_num, agent):
        self._action_list = range(0, action_num)
        self.agent = agent

    import random

    def get_max_action(self, game, i_agent, alpha=-float('inf'), beta=float('inf')):
        actions = list(range(0, game.get_action_num()))
        available_actions = [action for action in actions if game.external_check_action(action)]

        random.shuffle(available_actions)

        best_score = -float('inf')
        best_actions = []

        for action in available_actions:
            alt_game = deepcopy(game)
            _, reward, game_over, _ = alt_game.external_step(action)

            if game_over:
                if reward == 5 or reward == 0:
                    score = 0
                elif reward == 10:
                    score = 1
                else:
                    score = -1
            else:
                score = -self.get_max_action(alt_game, 1 if i_agent == 2 else 2, -beta, -alpha)[1]

            if score > best_score:
                best_score = score
                best_actions = [action]

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Alpha-beta pruning

        return torch.tensor(random.choice(best_actions)), best_score

    def select_action(self, env, action_sample):
        action, _ = self.get_max_action(env, self.agent)
        return action

    def learn(self, state, action, next_state, reward):
        pass
