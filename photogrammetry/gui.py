"""Tkinter 图形界面。

界面只负责交互与展示，全部计算委托给 :mod:`photogrammetry.solver`，
全部文本解析与输出委托给 :mod:`photogrammetry.io` / :mod:`photogrammetry.report`。
"""

from __future__ import annotations

import datetime
import math
import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from typing import Optional

from .io import (
    DataFormatError,
    format_points,
    load_example_data,
    read_points,
    timestamp_name,
    write_report,
)
from .report import build_report, format_summary
from .solver import Solution, SolveError, solve_exterior_orientation

APP_TITLE = "摄影测量外方位元素解算系统 v2.0"
DEFAULT_FOCAL = 152.0

FORMAT_HELP = """数据文件格式说明

1. 地面控制点（TXT/CSV）
   每行 3 个数值：X  Y  Z（通常为米），例如：
       100.25 299.75 45.12
       299.80 300.20 45.08

2. 像点坐标（TXT/CSV）
   每行 2 个数值：x  y（单位需与主距一致，通常为毫米），例如：
       -49.95 -50.02
        50.03 -49.98

3. 分隔符
   空格、制表符、逗号、分号均可；以 # 开头的整行或行尾内容视为注释。

4. 对应关系
   两个文件按行一一对应，点数必须相同，且至少需要 3 个不共线的控制点。
"""

ALGORITHM_HELP = """解算方法说明

· 模型：共线条件方程
    x - x0 = -f (a1ΔX + b1ΔY + c1ΔZ) / (a3ΔX + b3ΔY + c3ΔZ)
    y - y0 = -f (a2ΔX + b2ΔY + c2ΔZ) / (a3ΔX + b3ΔY + c3ΔZ)

· 初值：由地面点与像点的二维相似变换估计比例尺、旋角与摄影中心位置，
  再以控制点平均高程加航高作为 Zs 初值。

· 平差：Levenberg–Marquardt 阻尼最小二乘迭代，解析偏导，多组初值择优。

· 精度：m0 = sqrt(vᵀv / (2n - 6))，参数协方差 = m0² (AᵀA)⁻¹。

· 像点 y 轴：若对应关系为镜像（例如使用像素坐标，y 轴向下），
  程序可自动识别并翻转，报告中会注明。
"""


def _enable_dpi_awareness() -> None:
    """在 Windows 高 DPI 屏幕上保持界面清晰。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # pragma: no cover - 旧版 Windows 或权限受限
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class ToolTip:
    """轻量级按钮悬停提示。"""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tooltip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None) -> None:
        if self.tooltip is not None:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tooltip,
            text=self.text,
            background="#fffbe6",
            foreground="#333333",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=3,
        ).pack()

    def hide(self, _event=None) -> None:
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None


class PhotogrammetryApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        window_width, window_height = 1200, 840
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        left = max(0, (screen_width - window_width) // 2)
        top = max(0, (screen_height - window_height) // 3)
        self.root.geometry(f"{window_width}x{window_height}+{left}+{top}")
        self.root.minsize(1000, 700)

        self.ground = None
        self.image = None
        self.solution: Optional[Solution] = None
        self.ground_path: Optional[str] = None
        self.image_path: Optional[str] = None

        self.f_var = tk.StringVar(value=f"{DEFAULT_FOCAL:.1f}")
        self.x0_var = tk.StringVar(value="0.0")
        self.y0_var = tk.StringVar(value="0.0")
        self.yaxis_var = tk.StringVar(value="自动判断")
        self.ground_file_var = tk.StringVar(value="尚未导入")
        self.image_file_var = tk.StringVar(value="尚未导入")
        self.status_var = tk.StringVar(value="就绪：请导入地面控制点与像点坐标，或点击“加载示例”")

        self.ui_font, self.mono_font = self._resolve_fonts()
        self._configure_style()
        self._build_menu()
        self._build_toolbar()
        self._build_data_panes()
        self._build_notebook()
        self._build_statusbar()
        self._bind_shortcuts()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _resolve_fonts(self) -> tuple[str, str]:
        try:
            families = set(tkfont.families(self.root))
        except tk.TclError:  # pragma: no cover
            families = set()

        def pick(candidates: tuple[str, ...], fallback: str) -> str:
            return next((name for name in candidates if name in families), fallback)

        ui = pick(
            ("Microsoft YaHei UI", "微软雅黑", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"),
            "TkDefaultFont",
        )
        mono = pick(("Cascadia Mono", "Consolas", "Sarasa Mono SC", "DejaVu Sans Mono"), "Courier New")
        return ui, mono

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(self.ui_font, 10))
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", font=(self.ui_font, 10, "bold"))
        style.configure("Title.TLabel", font=(self.ui_font, 10, "bold"), foreground="#1f3a5f")
        style.configure("Hint.TLabel", font=(self.ui_font, 9), foreground="#6b7c8f")
        style.configure("Status.TLabel", font=(self.ui_font, 9), foreground="#34495e")
        style.configure("TNotebook.Tab", padding=(14, 6))
        style.configure("Treeview", font=(self.mono_font, 9), rowheight=24)
        style.configure("Treeview.Heading", font=(self.ui_font, 9, "bold"))

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="加载示例数据", accelerator="Ctrl+E", command=self.load_example)
        file_menu.add_command(label="导入地面控制点…", accelerator="Ctrl+O", command=self.load_ground)
        file_menu.add_command(label="导入像点坐标…", accelerator="Ctrl+I", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="导出解算报告…", accelerator="Ctrl+S", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.destroy)
        menubar.add_cascade(label="文件", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="数据格式说明", command=lambda: messagebox.showinfo("数据格式", FORMAT_HELP))
        help_menu.add_command(label="算法说明", command=lambda: messagebox.showinfo("算法说明", ALGORITHM_HELP))
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

    def _build_toolbar(self) -> None:
        frame = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        frame.pack(fill=tk.X)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)
        definitions = [
            ("加载示例", self.load_example, "载入 examples 目录中的示例数据（主距 152 mm）"),
            ("导入控制点", self.load_ground, "导入地面控制点 TXT/CSV（每行 X Y Z）"),
            ("导入像点", self.load_image, "导入像点坐标 TXT/CSV（每行 x y）"),
            ("开始解算", self.calculate, "按共线方程 + LM 迭代解算 6 个外方位元素"),
            ("导出报告", self.export_report, "导出包含精度与逐点残差的完整报告"),
            ("清空", self.clear_all, "清空已导入的数据与解算结果"),
        ]
        for column, (text, command, tip) in enumerate(definitions):
            style = "Accent.TButton" if text == "开始解算" else "TButton"
            button = ttk.Button(buttons, text=text, command=command, style=style)
            button.grid(row=0, column=column, padx=(0, 6))
            ToolTip(button, tip)

        camera = ttk.LabelFrame(frame, text="相机参数", padding=(10, 6))
        camera.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(camera, text="主距 f (mm)", style="Hint.TLabel").grid(row=0, column=0, padx=(0, 4))
        entry_f = ttk.Entry(camera, textvariable=self.f_var, width=9, justify=tk.RIGHT)
        entry_f.grid(row=0, column=1)
        ToolTip(entry_f, "相机主距，单位需与像点坐标一致；默认 152 mm")

        ttk.Label(camera, text="主点 x0 (mm)", style="Hint.TLabel").grid(row=0, column=2, padx=(18, 4))
        entry_x0 = ttk.Entry(camera, textvariable=self.x0_var, width=7, justify=tk.RIGHT)
        entry_x0.grid(row=0, column=3)
        ToolTip(entry_x0, "主点 x 方向偏移，默认 0")

        ttk.Label(camera, text="主点 y0 (mm)", style="Hint.TLabel").grid(row=0, column=4, padx=(18, 4))
        entry_y0 = ttk.Entry(camera, textvariable=self.y0_var, width=7, justify=tk.RIGHT)
        entry_y0.grid(row=0, column=5)
        ToolTip(entry_y0, "主点 y 方向偏移，默认 0")

        ttk.Label(camera, text="像点 y 轴", style="Hint.TLabel").grid(row=0, column=6, padx=(18, 4))
        combo = ttk.Combobox(
            camera,
            textvariable=self.yaxis_var,
            values=("自动判断", "向上", "向下"),
            width=9,
            state="readonly",
        )
        combo.grid(row=0, column=7)
        ToolTip(combo, "像素坐标通常 y 轴向下；自动判断会分析像点与地面点的对应关系")

        ttk.Label(
            camera,
            text="提示：主距与像点坐标单位需一致；控制点至少 3 个且不共线。",
            style="Hint.TLabel",
        ).grid(row=0, column=8, padx=(26, 0), sticky=tk.W)

    def _scrolled_text(
        self,
        parent: tk.Widget,
        height: int,
        wrap: str = tk.NONE,
        font: Optional[tuple] = None,
    ) -> tuple[ttk.Frame, tk.Text]:
        container = ttk.Frame(parent)
        text = tk.Text(
            container,
            height=height,
            wrap=wrap,
            font=font or (self.mono_font, 9),
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=6,
            background="#fbfcfd",
            foreground="#1f2d3d",
            insertbackground="#1f2d3d",
        )
        vbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text.yview)
        hbar = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        text.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        return container, text

    def _build_data_panes(self) -> None:
        panes = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=12, pady=(2, 6))
        self.main_panes = panes

        data_frame = ttk.Frame(panes, padding=(0, 2, 0, 2))
        panes.add(data_frame, weight=2)

        ground_frame = ttk.LabelFrame(data_frame, text="地面控制点 (X, Y, Z)", padding=6)
        ground_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(ground_frame, textvariable=self.ground_file_var, style="Hint.TLabel").pack(
            anchor=tk.W, pady=(0, 4)
        )
        ground_box, self.ground_text = self._scrolled_text(ground_frame, height=7)
        ground_box.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.LabelFrame(data_frame, text="像点坐标 (x, y)", padding=6)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(image_frame, textvariable=self.image_file_var, style="Hint.TLabel").pack(
            anchor=tk.W, pady=(0, 4)
        )
        image_box, self.image_text = self._scrolled_text(image_frame, height=7)
        image_box.pack(fill=tk.BOTH, expand=True)

        data_frame.columnconfigure(0, weight=1)
        data_frame.columnconfigure(1, weight=1)
        data_frame.rowconfigure(0, weight=1)

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.main_panes)
        self.main_panes.add(self.notebook, weight=3)

        # --- 解算结果 ---
        result_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(result_tab, text="解算结果")
        result_box, self.result_text = self._scrolled_text(result_tab, height=14)
        result_box.pack(fill=tk.BOTH, expand=True)
        self.result_text.configure(state=tk.DISABLED)

        # --- 残差与精度 ---
        residual_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(residual_tab, text="残差与精度")

        self.residual_note_var = tk.StringVar(value="")
        ttk.Label(residual_tab, textvariable=self.residual_note_var, style="Hint.TLabel").pack(
            anchor=tk.W, pady=(0, 4)
        )

        tree_frame = ttk.Frame(residual_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.residual_tree = ttk.Treeview(
            tree_frame,
            columns=("index", "x_obs", "y_obs", "x_calc", "y_calc", "vx", "vy", "r"),
            show="headings",
            height=7,
        )
        headings = {
            "index": ("点号", 60, "center"),
            "x_obs": ("x 观测", 110, "e"),
            "y_obs": ("y 观测", 110, "e"),
            "x_calc": ("x 计算", 110, "e"),
            "y_calc": ("y 计算", 110, "e"),
            "vx": ("vx", 100, "e"),
            "vy": ("vy", 100, "e"),
            "r": ("点位残差", 120, "e"),
        }
        for key, (text, width, anchor) in headings.items():
            self.residual_tree.heading(key, text=text)
            self.residual_tree.column(key, width=width, anchor=anchor, stretch=True)
        self.residual_tree.tag_configure("outlier", foreground="#c0392b")
        self.residual_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.residual_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.residual_tree.configure(yscrollcommand=scrollbar.set)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.precision_text = tk.Text(
            residual_tab,
            height=4,
            relief=tk.FLAT,
            borderwidth=0,
            padx=8,
            pady=6,
            wrap=tk.NONE,
            background="#f4f6f8",
            font=(self.mono_font, 9),
        )
        self.precision_text.pack(fill=tk.X, pady=(8, 0))
        self.precision_text.configure(state=tk.DISABLED)

        # --- 运行日志 ---
        log_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(log_tab, text="运行日志")
        log_box, self.log_text = self._scrolled_text(log_tab, height=10, wrap=tk.WORD)
        log_box.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

        self.root.after(120, self._set_sash)

    def _set_sash(self) -> None:
        try:
            height = self.main_panes.winfo_height()
            if height > 200:
                self.main_panes.sashpos(0, int(height * 0.36))
        except tk.TclError:  # pragma: no cover - 窗口尚未映射
            pass

    def _build_statusbar(self) -> None:
        status = ttk.Frame(self.root, padding=(12, 5))
        status.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            status,
            text="共线方程 · LM 平差",
            style="Hint.TLabel",
        ).pack(side=tk.RIGHT)

    def _bind_shortcuts(self) -> None:
        bindings = {
            "<Control-o>": self.load_ground,
            "<Control-i>": self.load_image,
            "<Control-e>": self.load_example,
            "<F5>": self.calculate,
            "<Control-s>": self.export_report,
            "<Control-l>": self.clear_all,
        }

        def make_handler(func):
            def handler(_event):
                func()
                return "break"

            return handler

        for sequence, handler in bindings.items():
            self.root.bind(sequence, make_handler(handler))

    # ------------------------------------------------------------------
    # 通用界面辅助
    # ------------------------------------------------------------------
    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.yview_moveto(0.0)
        widget.configure(state=tk.DISABLED)

    def log(self, message: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{stamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def show_error(self, message: str) -> None:
        self.status_var.set(f"错误：{message}")
        self.log(f"错误：{message}")
        messagebox.showerror("错误", message)

    def show_about(self) -> None:
        messagebox.showinfo(
            "关于",
            "摄影测量外方位元素解算系统 v2.0\n\n"
            "基于共线条件方程与 Levenberg–Marquardt 平差，\n"
            "解算单张影像的 6 个外方位元素。\n\n"
            "Python + NumPy + Tkinter",
        )

    # ------------------------------------------------------------------
    # 数据导入
    # ------------------------------------------------------------------
    @staticmethod
    def _ask_open(title: str) -> str:
        return filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("文本/CSV 文件", "*.txt *.csv"),
                ("文本文件", "*.txt"),
                ("CSV 文件", "*.csv"),
                ("所有文件", "*.*"),
            ],
        )

    def load_ground(self) -> None:
        path = self._ask_open("选择地面控制点文件")
        if not path:
            return
        try:
            data = read_points(path, 3, "地面控制点")
        except DataFormatError as exc:
            self.show_error(str(exc))
            return
        self.ground = data
        self.ground_path = path
        self.ground_file_var.set(f"{Path(path).name} · {len(data)} 个点")
        self._set_text(self.ground_text, format_points(data, "ground"))
        self.log(f"导入地面控制点：{path}（{len(data)} 个点）")
        self.status_var.set(f"已导入 {len(data)} 个地面控制点，请继续导入像点坐标")
        self._warn_count_mismatch()

    def load_image(self) -> None:
        path = self._ask_open("选择像点坐标文件")
        if not path:
            return
        try:
            data = read_points(path, 2, "像点坐标")
        except DataFormatError as exc:
            self.show_error(str(exc))
            return
        self.image = data
        self.image_path = path
        self.image_file_var.set(f"{Path(path).name} · {len(data)} 个点")
        self._set_text(self.image_text, format_points(data, "image"))
        self.log(f"导入像点坐标：{path}（{len(data)} 个点）")
        self.status_var.set(f"已导入 {len(data)} 个像点坐标，可点击“开始解算”")
        self._warn_count_mismatch()

    def _warn_count_mismatch(self) -> None:
        if self.ground is None or self.image is None:
            return
        if len(self.ground) != len(self.image):
            message = (
                f"控制点数量（{len(self.ground)}）与像点数量（{len(self.image)}）不一致，"
                "请检查两个文件是否逐行对应。"
            )
            self.status_var.set(message)
            self.log("警告：" + message)

    def load_example(self) -> None:
        try:
            ground, image, ground_path, image_path = load_example_data()
        except (DataFormatError, OSError) as exc:
            self.show_error(f"示例数据加载失败：{exc}")
            return

        self.ground, self.image = ground, image
        self.ground_path, self.image_path = str(ground_path), str(image_path)
        self.ground_file_var.set(f"{ground_path.name} · {len(ground)} 个点（示例）")
        self.image_file_var.set(f"{image_path.name} · {len(image)} 个点（示例）")
        self.f_var.set("152.0")
        self.x0_var.set("0.0")
        self.y0_var.set("0.0")
        self.yaxis_var.set("自动判断")
        self._set_text(self.ground_text, format_points(ground, "ground"))
        self._set_text(self.image_text, format_points(image, "image"))
        self.log(f"加载示例数据：{ground_path}，{image_path}")
        self.status_var.set("已加载示例数据（4 个控制点，f = 152 mm），点击“开始解算”")
        self._warn_count_mismatch()

    # ------------------------------------------------------------------
    # 解算与导出
    # ------------------------------------------------------------------
    def _camera_parameters(self) -> tuple[float, float, float]:
        entries = (("主距 f", self.f_var), ("x0", self.x0_var), ("y0", self.y0_var))
        values = []
        for name, variable in entries:
            text = variable.get().strip()
            try:
                value = float(text)
            except ValueError:
                raise SolveError(f"{name} 必须为数值") from None
            if not math.isfinite(value):
                raise SolveError(f"{name} 必须为有限数值")
            values.append(value)
        f, x0, y0 = values
        if f <= 0:
            raise SolveError("相机主距必须为正数")
        return f, x0, y0

    def calculate(self) -> None:
        if self.ground is None or self.image is None:
            self.show_error("请先导入地面控制点与像点坐标（或点击“加载示例”）")
            return

        try:
            f, x0, y0 = self._camera_parameters()
        except SolveError as exc:
            self.show_error(str(exc))
            return

        choice = self.yaxis_var.get()
        image_y_down = {"自动判断": None, "向上": False, "向下": True}.get(choice)

        self.status_var.set("正在解算外方位元素…")
        self.log(
            f"开始解算：{len(self.ground)} 个点，f={f:g}，x0={x0:g}，y0={y0:g}，"
            f"y 轴={choice}"
        )
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            solution = solve_exterior_orientation(
                self.ground,
                self.image,
                f=f,
                x0=x0,
                y0=y0,
                image_y_down=image_y_down,
            )
        except SolveError as exc:
            self.show_error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - 防御性兜底
            self.show_error(f"解算失败：{exc}")
            self.log(traceback.format_exc())
            return
        finally:
            self.root.configure(cursor="")

        self.solution = solution
        self._render_solution(solution)
        self.notebook.select(0)

    def _render_solution(self, solution: Solution) -> None:
        self._set_text(self.result_text, format_summary(solution))

        for row in self.residual_tree.get_children():
            self.residual_tree.delete(row)
        threshold = 3.0 * max(solution.rms, 1e-12)
        for index, (observed, computed, residual, point) in enumerate(
            zip(
                solution.observed,
                solution.computed,
                solution.residuals,
                solution.point_residuals,
            ),
            start=1,
        ):
            tags = ("outlier",) if point > threshold else ()
            self.residual_tree.insert(
                "",
                "end",
                values=(
                    index,
                    f"{observed[0]:.4f}",
                    f"{observed[1]:.4f}",
                    f"{computed[0]:.4f}",
                    f"{computed[1]:.4f}",
                    f"{residual[0]:+.4f}",
                    f"{residual[1]:+.4f}",
                    f"{point:.4f}",
                ),
                tags=tags,
            )

        precision_lines = [
            f"点位残差 RMS = {solution.rms:.6f}    坐标分量 RMS = {solution.rms_coordinate:.6f}"
            f"    最大点位残差 = {solution.max_residual:.6f}"
            + (
                f"    单位权中误差 σ0 = {solution.sigma0:.6f}"
                if solution.sigma0 is not None
                else ""
            ),
            (
                f"法方程条件数 = {solution.condition_number:.4g}    "
                f"迭代 {solution.iterations} 次"
                f"（{'收敛' if solution.converged else '未严格收敛'}）"
            ),
        ]
        if solution.std:
            precision_lines.append(
                "参数精度："
                f"Xs ±{solution.std.get('Xs', float('nan')):.3f} m，"
                f"Ys ±{solution.std.get('Ys', float('nan')):.3f} m，"
                f"Zs ±{solution.std.get('Zs', float('nan')):.3f} m；"
                f"φ ±{solution.std.get('phi', float('nan')):.6f}°，"
                f"ω ±{solution.std.get('omega', float('nan')):.6f}°，"
                f"κ ±{solution.std.get('kappa', float('nan')):.6f}°"
            )
        self._set_text(self.precision_text, "\n".join(precision_lines) + "\n")
        self.residual_note_var.set(
            "注：像点 y 轴已按像素坐标系（向下）翻转，下表为翻转后的坐标与残差。"
            if solution.image_y_flipped
            else "下表为像点坐标观测值、计算值与残差（与输入文件坐标系一致）。"
        )

        for warning in solution.warnings:
            self.log("提示：" + warning)
        self.status_var.set(
            f"解算完成：Xs={solution.xs:.3f}，Ys={solution.ys:.3f}，Zs={solution.zs:.3f}，"
            f"RMS={solution.rms:.4g}"
        )
        self.log(
            f"解算完成：Xs={solution.xs:.4f}，Ys={solution.ys:.4f}，Zs={solution.zs:.4f}，"
            f"φ={solution.phi:.6f}°，ω={solution.omega:.6f}°，κ={solution.kappa:.6f}°，"
            f"RMS={solution.rms:.6g}，迭代 {solution.iterations} 次"
        )

    def export_report(self) -> None:
        if self.solution is None:
            self.show_error("暂无解算结果，请先完成解算")
            return
        path = filedialog.asksaveasfilename(
            title="导出解算报告",
            defaultextension=".txt",
            initialfile=timestamp_name(),
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            report = build_report(
                self.solution,
                ground_path=self.ground_path,
                image_path=self.image_path,
            )
            write_report(path, report)
        except OSError as exc:
            self.show_error(f"报告写入失败：{exc}")
            return
        self.log(f"报告已导出：{path}")
        self.status_var.set(f"报告已导出：{path}")
        messagebox.showinfo("导出成功", f"解算报告已保存至：\n{path}")

    def clear_all(self) -> None:
        self.ground = None
        self.image = None
        self.solution = None
        self.ground_path = None
        self.image_path = None
        self.ground_file_var.set("尚未导入")
        self.image_file_var.set("尚未导入")
        self.f_var.set(f"{DEFAULT_FOCAL:.1f}")
        self.x0_var.set("0.0")
        self.y0_var.set("0.0")
        self.yaxis_var.set("自动判断")
        for widget in (self.ground_text, self.image_text):
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
        self._set_text(self.result_text, "")
        self._set_text(self.precision_text, "")
        self._set_text(self.log_text, "")
        self.residual_note_var.set("")
        for row in self.residual_tree.get_children():
            self.residual_tree.delete(row)
        self.status_var.set("就绪：已清空数据，请重新导入")
        self.log("已清空所有数据")


def main(argv: Optional[list] = None) -> None:
    """启动图形界面；``--demo`` 会在启动后自动载入示例并解算。"""
    import argparse

    parser = argparse.ArgumentParser(description="摄影测量外方位元素解算系统")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="启动后自动加载 examples 中的示例数据并执行解算",
    )
    args = parser.parse_args(argv)

    _enable_dpi_awareness()
    root = tk.Tk()
    app = PhotogrammetryApp(root)
    if args.demo:

        def run_demo() -> None:
            app.load_example()
            app.calculate()

        root.after(150, run_demo)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
