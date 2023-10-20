import random
import tkinter as tk
import tkinter.font as font
from tkinter import SUNKEN

import numpy as np


class SquareButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        side = kwargs.pop('side_length', None)
        tk.Button.__init__(self, master, compound='center', **kwargs)
        if side:
            self.config(width=side, height=side)


class TicTacToeGame:
    def __init__(self):
        self._size = 3
        self._win_size = 3
        self._grid_field = [[0 for x in range(self._size)] for y in range(self._size)]
        self._root = None
        self._current_player = 1

    def _set_grid_field_value(self, row, col, value):
        self._grid_field[row][col] = value

    def _toggle_current_player(self):
        self._current_player = 2 if self._current_player == 1 else 1

    def create_env(self):
        self._root = tk.Tk()
        self._root.title("Tic tac toe")
        self._root.geometry("600x600")
        self.draw_field()

    def set_text(self, button, r, c):
        current_field = self._grid_field[r][c]
        button_text = ""
        if current_field == 1:
            button_text = "❌"
        elif current_field == 2:
            button_text = "⭕"

        button.configure(text=button_text)

    def set_button(self, r, c):
        btn = SquareButton(side_length=200, font=font.Font(size=75), background="black", fg="white",
                           activebackground="black", activeforeground="white", relief=SUNKEN)
        self.set_text(btn, r, c)
        btn.grid(row=r, column=c)

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
                    return True, next(iter(row_arr))

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
        self._grid_field = [[0 for x in range(self._size)] for y in range(self._size)]

        if not is_external:
            self.draw_field()
        return np.array(self._grid_field).flatten()

    def get_action_num(self):
        return self._size * self._size

    def get_action_sample(self):
        return random.choice(range(0, self.get_action_num()))

    def step(self, r, c, is_external=False):
        if not self.is_empty(r, c):
            return np.array(self._grid_field).flatten(), -10, False, False

        self._set_grid_field_value(r, c, self._current_player)

        if not is_external:
            self.set_button(r, c)

        self._toggle_current_player()

        has_ended, winner = self.has_ended()
        if has_ended:
            self.reset(is_external)
            return np.array(self._grid_field).flatten(), 10 if winner != 0 else 0, True, False

        return np.array(self._grid_field).flatten(), 0, False, False

    def on_click(self, event):
        btn_info = event.widget.grid_info()
        self.step(btn_info['row'], btn_info['column'])

    def external_step(self, action):
        return self.step(action // self._size, action % self._size, True)

    def draw_field(self, is_replay_mode=False):
        for c in range(self._size):
            self._root.columnconfigure(index=c, weight=1)
        for r in range(self._size):
            self._root.rowconfigure(index=r, weight=1)

        for r in range(self._size):
            for c in range(self._size):
                self.set_button(r, c)

        if is_replay_mode is False:
            self._root.bind("<1>", self.on_click)
            self._root.mainloop()


if __name__ == '__main__':
    tg = TicTacToeGame()
    tg.create_env()
