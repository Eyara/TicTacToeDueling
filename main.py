import hashlib
import json
import random

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from agents.minimax_agent import MiniMaxAgent
from board_gui import BoardGUI


class TicTacToeGame:
    def __init__(self):
        self._size = 3
        self._win_size = 3
        self._grid_field = np.zeros((self._size, self._size))
        self._root = None
        self._current_player = 1

        self._board = BoardGUI(self._size)
        self._board.on("step", self.step_event_handler)
        self._board.on("agent_step", self.agent_step_event_handler)

        self._x_agent = DQNAgent("cuda", self.get_state(), self.get_action_num(), "x")
        # self._x_agent = MiniMaxAgent(self.get_action_num(), 1)
        self.is_against_agent = False

    def _set_grid_field_value(self, row, col, value):
        self._grid_field[row][col] = value

    def _toggle_current_player(self):
        self._current_player = 2 if self._current_player == 1 else 1

    def get_current_player(self):
        return self._current_player

    def set_grid_field(self, grid):
        self._grid_field = grid

    def is_empty(self, r, c):
        return self._grid_field[r][c] == 0

    def has_ended(self):
        iteration_size = self._size - self._win_size
        # row  and column check
        for i in range(0, self._size):
            for j in range(0, iteration_size + 1):
                row_arr = set()
                col_arr = set()

                for k in range(0, self._win_size):
                    row_arr.add(self._grid_field[i][j + k])
                    col_arr.add(self._grid_field[j + k][i])

                if len(row_arr) == 1 and next(iter(row_arr)) != 0:
                    return True, next(iter(row_arr))
                if len(col_arr) == 1 and next(iter(col_arr)) != 0:
                    return True, next(iter(col_arr))

        # diagonal check
        for i in range(0, iteration_size + 1):
            for j in range(0, iteration_size + 1):
                main_arr = set()
                side_arr = set()

                for k in range(0, self._win_size):
                    main_arr.add(self._grid_field[k][k])
                    side_arr.add(self._grid_field[k][self._size - k - 1])

                if len(main_arr) == 1 and next(iter(main_arr)) != 0:
                    return True, next(iter(main_arr))
                if len(side_arr) == 1 and next(iter(side_arr)) != 0:
                    return True, next(iter(side_arr))

        if not any(0 in row for row in self._grid_field):
            return True, 0

        return False, None

    def reset(self, is_external=False):
        self._grid_field = np.zeros((self._size, self._size))
        self._current_player = 1

        if not is_external:
            self._board.draw_field()
        return np.array(self._grid_field).flatten()

    def get_action_num(self):
        return self._size * self._size

    def get_action_sample(self):
        return random.choice(range(0, self.get_action_num()))

    def get_grid_field(self):
        return self._grid_field

    def get_state(self):
        return torch.tensor(np.array(self._grid_field).flatten(), dtype=torch.float32, device="cuda").unsqueeze(0)

    def step(self, r, c, is_external=False):
        if not self.is_empty(r, c):
            return np.array(self._grid_field).flatten(), -2, False, False

        self._set_grid_field_value(r, c, self._current_player)

        if not is_external:
            self._board.set_button(r, c, self._grid_field[r][c])

        self._toggle_current_player()

        has_ended, winner = self.has_ended()
        if has_ended:
            ended_grid_field = self._grid_field
            return np.array(ended_grid_field).flatten(), 10 if winner != 0 else 5, True, False

        return np.array(self._grid_field).flatten(), 0, False, False

    def external_step(self, action):
        state, reward, done, truncated = self.step(action // self._size, action % self._size, True)
        if done:
            self.reset(True)
        return state, reward, done, truncated

    def external_check_action(self, action):
        return self.is_empty(action // self._size, action % self._size)

    def set_play_agent_mode(self):
        self.is_against_agent = True

    def agent_step(self):
        while True:
            action = self._x_agent.select_action(self.get_state(), self.get_action_sample())
            # copy_game = TicTacToeGame()
            # copy_game.set_grid_field(self._grid_field)
            # action = self._x_agent.select_action(copy_game, self.get_action_sample())
            action = action.item()

            if self.external_check_action(action):
                state, reward, done, truncated = self.step(action // self._size, action % self._size)
                print(reward)
                if done:
                    self.reset()
                break

    def step_event_handler(self, data):
        state, reward, done, truncated = self.step(data[0], data[1])
        if done:
            self.reset()

    def agent_step_event_handler(self, data):
        if self.is_against_agent:
            self.agent_step()

    def create_env(self):
        self._board.create_env()

    def create_replay_env(self):
        self._board.create_replay_env()

    def run_mainloop(self):
        self._board.run_mainloop()


if __name__ == '__main__':
    tg = TicTacToeGame()
    tg.set_play_agent_mode()
    tg.create_env()
