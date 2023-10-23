import random
from copy import deepcopy

import torch

from main import TicTacToeGame


class MiniMaxAgent:
    def __init__(self, action_num):
        self._action_list = range(0, action_num)
        self.cache = {}
        self.cache_other = {}

    def get_from_cache(self, game: TicTacToeGame, i_agent: int):
        hash = game.get_hash()
        if i_agent == 2 and hash in self.cache:
            actions, score = self.cache[hash]
            return torch.tensor(random.choice(actions)), score
        elif hash in self.cache_other:
            actions, score = self.cache_other[hash]
            return torch.tensor(random.choice(actions)), score
        return None

    def store_in_cache(self, game: TicTacToeGame, i_agent: int, actions, score: int):
        hash = game.get_hash()
        if i_agent == 2:
            self.cache[hash] = (actions, score)
        else:
            self.cache_other[hash] = (actions, score)

    def get_max_action(self, game: TicTacToeGame, i_agent):
        from_cache = self.get_from_cache(game, i_agent)
        if not from_cache is None:
            return from_cache

        actions = list(range(0, game.get_action_num()))
        available_actions = []
        for action in actions:
            if game.external_check_action(action):
                available_actions.append(action)

        random.shuffle(actions)

        best_score = None
        best_actions = []
        for action in available_actions:
            alt_game = deepcopy(game)
            _, reward, game_over, _ = alt_game.external_step(action)

            score = 0
            if game_over:
                if reward == 5 or reward == 0:
                    score = 0
                elif reward == 10:
                    score = 1
                else:
                    score = -1
            else:
                score = self.get_max_action(alt_game, game.get_current_player())[1] * -1

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        self.store_in_cache(game, i_agent, best_actions, best_score)
        return torch.tensor(random.choice(best_actions)), best_score

    def select_action(self, env, action_sample):
        action, _ = self.get_max_action(env, 2)
        return action

    def learn(self, state, action, next_state, reward):
        # Random Agent cant learn :(
        pass
