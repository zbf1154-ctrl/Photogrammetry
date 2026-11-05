"""程序入口：``python main.py`` 启动图形界面。

核心算法位于 :mod:`photogrammetry.solver`，也可以直接在脚本中调用::

    from photogrammetry import solve_exterior_orientation

    solution = solve_exterior_orientation(ground, image, f=152.0)
    print(solution.params_deg)
"""

from __future__ import annotations

from photogrammetry.gui import main

if __name__ == "__main__":
    main()
