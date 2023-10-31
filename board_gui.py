import tkinter as tk
from tkinter import SUNKEN, font


class SquareButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        side = kwargs.pop('side_length', None)
        tk.Button.__init__(self, master, compound='center', **kwargs)
        if side:
            self.config(width=side, height=side)


class BoardGUI:
    def __init__(self, size):
        self._size = size
        self._root = None
        self.listeners = {}
        self.step_event_name = "step"
        self.step_agent_event_name = "agent_step"

    def get_root(self):
        return self._root

    def create_replay_env(self):
        self._root = tk.Tk()
        self._root.title("Tic tac toe's replay")
        self._root.geometry("600x600")

    def run_mainloop(self):
        self._root.mainloop()

    def create_env(self):
        self._root = tk.Tk()
        self._root.title("Tic tac toe")
        self._root.geometry("600x600")
        self.draw_field()

    @staticmethod
    def set_text(button, grid_value):
        button_text = ""
        if grid_value == 1:
            button_text = "❌"
        elif grid_value == 2:
            button_text = "⭕"

        button.configure(text=button_text)

    def set_button(self, r, c, grid_value):
        btn = SquareButton(side_length=200, font=font.Font(size=75), background="black", fg="white",
                           activebackground="black", activeforeground="white", relief=SUNKEN)
        self.set_text(btn, grid_value)
        btn.grid(row=r, column=c)

    def draw_replay_field(self, grid):
        for c in range(self._size):
            self._root.columnconfigure(index=c, weight=1)
        for r in range(self._size):
            self._root.rowconfigure(index=r, weight=1)

        for r in range(self._size):
            for c in range(self._size):
                self.set_button(r, c, grid[r][c])

    def draw_field(self):
        for c in range(self._size):
            self._root.columnconfigure(index=c, weight=1)
        for r in range(self._size):
            self._root.rowconfigure(index=r, weight=1)

        for r in range(self._size):
            for c in range(self._size):
                self.set_button(r, c, 0)

        self.emit(self.step_agent_event_name)

        self._root.bind("<1>", self.on_click)
        self._root.mainloop()

    def on_click(self, event):
        btn_info = event.widget.grid_info()
        self.emit(self.step_event_name, (btn_info['row'], btn_info['column']))
        self.emit(self.step_agent_event_name)

    def on(self, event_name, callback):
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(callback)

    def emit(self, event_name, data=None):
        if event_name in self.listeners:
            for callback in self.listeners[event_name]:
                callback(data)
