import matplotlib.pyplot as plt
import numpy as np
import glob


def read_file(filename):

    f = open(filename)
    lines = f.readlines()

    array = []
    for li in lines:
        li = li.split()
        li = [float(l) for l in li]
        array.append(li)

    return array


def plot_quantity(file_regexp):

    policy_filenames = glob.glob(file_regexp)
    policy_filenames.sort()

    print(policy_filenames)

    for pf in policy_filenames:
        array = read_file(pf)

        plt.imshow(np.flipud(array), interpolation='none', aspect='equal', origin='lower')

        ax = plt.gca()

        ax.set_xticks(np.arange(0, 21, 1))
        ax.set_yticks(np.arange(0, 21, 1))

        ax.set_xticklabels(np.arange(0, 21, 1))
        ax.set_yticklabels(np.arange(0, 21, 1))

        ax.set_xticks(np.arange(-.5, 21, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 21, 1), minor=True)

        # plt.pcolormesh(array, edgecolors='k', linewidth=2)
        # ax = plt.gca()
        # ax.set_aspect('equal')

        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)


        plt.show()

if __name__ == "__main__":

    plot_quantity("policy_iter_*txt")
    plot_quantity("value_iter_*txt")
