from __future__ import annotations

from pathlib import Path

from model import LaunchModel
from plot import plot_one


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "_1_InitialDataSV3.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(
            "Поруч з main.py немає _1_InitialDataSV3.xlsx. "
            "Без цього файла Python-порт не зможе відтворити чисельно той самий розрахунок."
        )

    model = LaunchModel(excel_path=excel_path, unit_system="SI")
    result = model.simulate()
    print(f"Saved points: {len(result.time)}")
    plot_one(result.history, "omega", base_dir / "omega.png")


if __name__ == "__main__":
    main()
