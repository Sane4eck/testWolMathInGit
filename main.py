# main.py
import os, argparse

ap = argparse.ArgumentParser()
# ap.add_argument("--system", default="sys_C_0")
ap.add_argument("--system", default="sys_with_C")
args = ap.parse_args()

os.environ["DYNAMICS_SYSTEM"] = args.system


from core.state import State
from core.system import Params, initial_y
from core.model import HydraulicModel
from core.result import Result
import time

dt = 1e-7
endTime = 1.0

state = State(time=0.0, y=initial_y())
params = Params()
model = HydraulicModel()

t0 = time.perf_counter()
t_arr, y_arr, dy_arr, aux_arr = model.simulate(state, params, dt, endTime, countPoint=3000, backend="numba")
res = Result(t_arr, y_arr, dy_arr, aux_arr)
t1 = time.perf_counter()
print(f"Execution time: {t1-t0:.2f} s")

from plot import plot_results
plot_results(res.data)

import time
run_id = time.strftime("%Y%m%d_%H%M%S")

from core.plotting import save_all_plots
save_all_plots(res, out_dir=f"out/plots/{run_id}", show=False)

from core.export_excel import save_to_excel
save_to_excel(res, f"out/data/{run_id}.xlsx")
