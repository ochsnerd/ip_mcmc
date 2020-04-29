import matplotlib.pyplot as plt


def store_figure(name):
    """
    Store a figure in the figures directory.
    Assumes there is an active pyplot-Plot and clears it after
    """
    plt.savefig("../figures/" + name + ".svg", format='svg')
    plt.clf()
