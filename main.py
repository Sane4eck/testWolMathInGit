# main.py
import os
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="sys_wolmath")
ap.add_argument("--excel", default="_1_InitialDataSV3.xlsx")
ap.add_argument("--dt", type=float, default=1e-7)
ap.add_argument("--end-time", type=float, default=1.0)
args = ap.parse_args()

os.environ["DYNAMICS_SYSTEM"] = args.system

from core.state import State
from core.system import Params, initial_y
from core.model import HydraulicModel
from core.result import Result
from plot import plot_results
from core.plotting import save_all_plots
from core.export_excel import save_to_excel

def main():
    params = Params.from_excel(args.excel)
    state = State(time=0.0, y=initial_y(params))
    model = HydraulicModel()

    t0 = time.perf_counter()
    t_arr, y_arr, dy_arr, aux_arr = model.simulate(
        state, params, args.dt, args.end_time, countPoint=3000, backend="numba"
    )
    res = Result(t_arr, y_arr, dy_arr, aux_arr)
    t1 = time.perf_counter()

    print(f"Execution time: {t1 - t0:.2f} s")

    plot_results(res.data)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_all_plots(res, out_dir=f"out/plots/{run_id}", show=False)
    save_to_excel(res, f"out/data/{run_id}.xlsx")

if __name__ == "__main__":
    main()
