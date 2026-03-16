# plot.py
import os
import matplotlib.pyplot as plt

def plot_results(result, save_dir=None, prefix="run", show=True):
    t  = [r["time"] for r in result]
    p0 = [r["p0"]   for r in result]
    p1 = [r["p1"]   for r in result]
    m01= [r["m01"]  for r in result]
    m12= [r["m12"]  for r in result]
    m13= [r["m13"]  for r in result]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Figure 1: pressures
    fig1 = plt.figure()
    plt.plot(t, p0, label="p0")
    plt.plot(t, p1, label="p1")
    plt.xlabel("t, s")
    plt.ylabel("p, bar")
    plt.grid()
    plt.legend()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, f"{prefix}_p.png"), dpi=200, bbox_inches="tight")

    # Figure 2: flows
    fig2 = plt.figure()
    plt.plot(t, m01, label="m01")
    plt.plot(t, m12, label="m12")
    plt.plot(t, m13, label="m13")
    plt.xlabel("t, s")
    plt.ylabel("m")
    plt.grid()
    plt.legend()
    if save_dir:
        fig2.savefig(os.path.join(save_dir, f"{prefix}_m.png"), dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
