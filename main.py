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
        self._grid_field = [[0 for x in range(self._size)] for y in range(self._size)]
        self._grid_field[1][1] = 1
        self._grid_field[1][0] = 2
        self._root = None

    def _set_grid_field_value(self, row, col, value):
        self._grid_field[row][col] = value
        self.set_button(row, col)

    def create_env(self):
        self._root = tk.Tk()
        self._root.title("Tic tac toe")
        self._root.geometry("600x600")
        self.draw_field()

    def set_text(self, button, r, c):
        if self._grid_field[r][c] == 1:
            button.configure(text="X")
        elif self._grid_field[r][c] == 2:
            button.configure(text="O")
        else:
            button.configure(text="")

    def set_button(self, r, c):
        btn = SquareButton(side_length=200, font=font.Font(size=125), background="black", fg="white",
                           activebackground="black", activeforeground="white", relief=SUNKEN)
        self.set_text(btn, r, c)
        btn.grid(row=r, column=c)

    def on_click(self, event):
        btn_info = event.widget.grid_info()
        self._set_grid_field_value(btn_info['row'], btn_info['column'], 1)

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
