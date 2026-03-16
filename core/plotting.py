# core/plotting.py
import os
import matplotlib.pyplot as plt

from core.system import Y_ORDER, AUX_ORDER

def _is_pressure(name: str) -> bool:
    # p*, dp* конвертуємо в bar (або bar/s)
    return name.startswith("p") or name.startswith("dp") or name.startswith("dP")

def save_all_plots(res, out_dir="out/plots", show=False):
    os.makedirs(out_dir, exist_ok=True)

    t = res.t

    # y
    for j, name in enumerate(Y_ORDER):
        y = res.y[:, j]
        ylabel = name
        yplot = y * 1e-5 if _is_pressure(name) else y
        if _is_pressure(name):
            ylabel += " [bar]"
        fig = plt.figure()
        plt.plot(t, yplot)
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel(ylabel)
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # aux
    for j, name in enumerate(AUX_ORDER):
        a = res.aux[:, j]
        ylabel = name
        aplot = a * 1e-5 if _is_pressure(name) else a
        if _is_pressure(name):
            ylabel += " [bar]"
        fig = plt.figure()
        plt.plot(t, aplot)
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel(ylabel)
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # dy
    dy_names = tuple("d" + n for n in Y_ORDER)
    for j, name in enumerate(dy_names):
        d = res.dy[:, j]
        ylabel = name
        dplot = d * 1e-5 if _is_pressure(name) else d
        if _is_pressure(name):
            ylabel += " [bar/s]"
        fig = plt.figure()
        plt.plot(t, dplot)
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel(ylabel)
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
