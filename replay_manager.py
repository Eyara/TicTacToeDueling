import itertools

import numpy as np

from main import TicTacToeGame


class ReplayManager:
    def __init__(self):
        self.filename = 'train_states.txt'

    def save_to_file(self, training_states):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        with open('./%s' % self.filename, 'w') as f:
            cur_episode = -1

            for i in training_states:
                if cur_episode < i[0]:
                    cur_episode += 1
                    f.write('Episode: %s \n' % cur_episode)

                f.write(np.array2string(np.array(i[1]), separator=', '))
                f.write('\n')

    def load_from_file(self):
        result = []
        with open('./%s' % self.filename, 'r') as f:
            cur_episode = -1

            for line in f:
                if 'Episode' in line:
                    cur_episode += 1
                else:
                    result.append((cur_episode, [int(i) for i in
                                                 line.replace('\n', '').replace('[', '').replace(']', '').split(', ')]))

        return [list(g) for k, g in itertools.groupby(result, lambda x: x[0])]

    def get_top_n(self, num):
        result = self.load_from_file()
        return sorted(filter(lambda x: len(x) > 3, result), key=lambda x: len(x))[:num]


def play_replay(episode, env, root):
    for step in episode[0]:
        env.set_grid_field(np.array(step[1]).reshape((3, 3)))
        env.draw_field(True)
        root.after(750)
        root.update()


if __name__ == '__main__':
    rm = ReplayManager()
    best_episode = rm.get_top_n(1)

    env = TicTacToeGame()
    env.create_replay_env()

    play_replay(best_episode, env, env.get_root())

    env.run_mainloop()
