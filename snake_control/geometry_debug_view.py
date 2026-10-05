"""显示 GeometryExchange 实际收到的曲线；不连接网络、不参与控制。"""
import math
import multiprocessing as mp
import queue
import time


def segment_points(segment):
    """按可视化端的直线/有向圆弧定义采样，坐标和半径单位 mm。"""
    x, y = segment["start_mm"]
    tx, ty = segment["tangent"]
    radius, turn = segment["radius_mm"], segment["turn_rad"]
    if segment["kind"] == "line" or radius == 0:
        length = segment["length_mm"]
        return [(x, y), (x + tx * length, y + ty * length)]
    count = max(2, min(720, math.ceil(abs(turn) / math.radians(3))))
    signed_radius = math.copysign(radius, turn)
    points = []
    for i in range(count + 1):
        angle = turn * i / count
        along = signed_radius * math.sin(angle)
        normal = signed_radius * 2 * math.sin(angle / 2) ** 2
        points.append((x + tx * along - ty * normal, y + ty * along + tx * normal))
    return points


class GeometryDebugWindow:
    COLORS = ("#2676c9", "#d56a22", "#249176", "#9258b2", "#c14967", "#71812c")

    def __init__(self, root, inbox, draw_curve=True):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.inbox = root, inbox
        self.packet = None
        self.segments = []
        self.paths = []
        root.title("Adaptive Pole · 连续曲线接收调试")
        root.geometry("1120x650" if draw_curve else "520x650")
        root.minsize(900 if draw_curve else 500, 450)
        self.status = tk.Label(root, text="等待可视化程序回传曲线……", anchor="w",
                               padx=14, pady=12, font=("Microsoft YaHei UI", 11))
        self.status.pack(fill="x")
        content = ttk.Frame(root, padding=(12, 0, 12, 8))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.canvas = None
        if draw_curve:
            self.canvas = tk.Canvas(content, background="#f7f9fc", highlightthickness=0)
            self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        panel = ttk.Frame(content)
        panel.grid(row=0, column=1 if draw_curve else 0, sticky="nsew")
        columns = ("segment", "joint", "kind", "radius", "turn", "length")
        self.table = ttk.Treeview(panel, columns=columns, show="headings", selectmode="browse")
        for name, title, width in zip(columns,
                ("段", "关节", "类型", "半径 mm", "圆心角 °", "段长 mm"),
                (40, 45, 50, 85, 85, 85)):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, anchor="center", stretch=False)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")
        hint = "可视化程序 → Python 实际反馈  |  圆心角保留转向正负号"
        if draw_curve:
            hint += "  |  点击表格高亮曲线段"
        ttk.Label(root, text=hint,
                  padding=(14, 4, 14, 10)).pack(anchor="w")
        if self.canvas is not None:
            self.canvas.bind("<Configure>", lambda event: self.draw())
        self.table.bind("<<TreeviewSelect>>", lambda event: self.draw())
        root.after(50, self.poll)

    def poll(self):
        try:
            while True:
                packet = self.inbox.get_nowait()
                if packet is None:
                    self.root.destroy()
                    return
                self.set_packet(packet)
        except queue.Empty:
            pass
        self.update_status()
        self.root.after(100, self.poll)

    def set_packet(self, packet):
        self.packet = packet
        self.segments = packet.get("segments", [])
        self.paths = ([segment_points(segment) for segment in self.segments]
                      if self.canvas is not None else [])
        current_ids = set()
        for segment in self.segments:
            index = segment["segment_index"]
            iid = str(index)
            current_ids.add(iid)
            arc = segment["kind"] == "arc"
            values = (index + 1, segment["joint_index"] or "端部", "圆弧" if arc else "直线",
                      f'{segment["radius_mm"]:.2f}' if arc else "—",
                      f'{math.degrees(segment["turn_rad"]):+.2f}' if arc else "—",
                      f'{segment["length_mm"]:.2f}')
            if self.table.exists(iid):
                self.table.item(iid, values=values)
            else:
                self.table.insert("", "end", iid=iid, values=values)
        for iid in self.table.get_children():
            if iid not in current_ids:
                self.table.delete(iid)
        self.update_status()
        self.draw()

    def update_status(self):
        packet = self.packet
        if not packet or packet.get("sequence") is None:
            self.status.configure(text="未收到新鲜反馈：等待连接或恢复数据", fg="#a34d00")
            return
        age = packet["age_ms"] + (time.monotonic() - packet["sent_at"]) * 1000
        if age > 500:
            label, color = "反馈已过期（画面保留最后一帧）", "#a34d00"
        elif not packet["valid"]:
            label, color = "收到反馈，曲线无效", "#b3261e"
        else:
            label, color = "正在接收 · 曲线有效", "#167347"
        self.status.configure(text=f'{label}  |  帧 {packet["sequence"]}  |  {len(self.segments)} 段',
                              fg=color)

    def draw(self):
        canvas = self.canvas
        if canvas is None:
            return
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.create_text(16, 18, anchor="nw", text="二维连续曲线（等比例，单位 mm）", fill="#526075")
        points = [point for path in self.paths for point in path]
        if not points or width < 100 or height < 100:
            return
        xs, ys = zip(*points)
        scale = min((width - 90) / max(max(xs) - min(xs), 1),
                    (height - 100) / max(max(ys) - min(ys), 1))
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

        def screen(point):
            return width / 2 + (point[0] - cx) * scale, height / 2 - (point[1] - cy) * scale

        selected = self.table.selection()
        for segment, path in zip(self.segments, self.paths):
            index = segment["segment_index"]
            chosen = str(index) in selected
            color = "#e62323" if chosen else self.COLORS[index % len(self.COLORS)]
            coords = [value for point in path for value in screen(point)]
            item = canvas.create_line(*coords, fill=color, width=6 if chosen else 3,
                                      tags=("curve", f"segment-{index}"))
            if chosen:
                canvas.tag_raise(item)
                px, py = screen(path[len(path) // 2])
                canvas.create_text(px, py - 15, text=f"段 {index + 1}", fill=color)
        for point, label in ((points[0], "头"), (points[-1], "尾")):
            px, py = screen(point)
            canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#263449", outline="")
            canvas.create_text(px + 12, py + 12, text=label, fill="#263449")
        # 以当前比例显示一条有单位的标尺。
        length = 10 ** math.floor(math.log10(80 / scale))
        canvas.create_line(20, height - 25, 20 + length * scale, height - 25, width=2)
        canvas.create_text(20, height - 30, anchor="sw", text=f"{length:g} mm")


def _run_window(inbox, draw_curve):
    import tkinter as tk
    root = tk.Tk()
    GeometryDebugWindow(root, inbox, draw_curve)
    root.mainloop()


class GeometryDebugView:
    """GUI 独立运行；主循环每 200 ms 非阻塞地投递一次最新反馈。"""

    def __init__(self, draw_curve=True):
        context = mp.get_context("spawn")
        self.inbox = context.Queue(maxsize=1)
        self.process = context.Process(target=_run_window, args=(self.inbox, draw_curve), daemon=True)
        self.process.start()
        self.next_update = 0.0

    def update(self, exchange):
        now = time.monotonic()
        if now < self.next_update or not self.process.is_alive():
            return
        self.next_update = now + 0.2
        geometry = exchange.latest_geometry() if exchange is not None else None
        packet = {"sent_at": now}
        if geometry is not None:
            curve = geometry["curve"]
            packet.update(sequence=geometry["sequence"], age_ms=geometry["age_ms"],
                          valid=curve["valid"], segments=curve["segments"])
        self._offer(packet)

    def _offer(self, packet):
        try:
            self.inbox.put_nowait(packet)
        except queue.Full:
            pass  # 窗口忙时跳过显示帧，绝不等待绘图。

    def close(self):
        self._offer(None)
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.inbox.cancel_join_thread()
        self.inbox.close()
