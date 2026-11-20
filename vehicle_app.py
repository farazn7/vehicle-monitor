"""Desktop controls and event history for the vehicle monitor."""
import argparse
from collections import deque
from contextlib import closing
import os
from pathlib import Path
import queue
import sqlite3
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from vehicle_event_log import EmailSettings, initialize_database
from zone_monitor import (
    DATABASE, DEFAULT_ZONE, PARKING_VIDEO, PARKING_ZONE, VIDEO,
    MonitorOptions, preset_zone, process_video, read_first_frame,
)


class VehicleApp:
    def __init__(self, root, weights, device):
        self.root = root
        self.weights = weights
        self.device = device
        self.video = None
        self.first_frame = None
        self.preview_frame = None
        self.photo = None
        self.zone = None
        self.points = []
        self.drawing = False
        self.image_bounds = None

        self.worker = None
        self.stop_requested = threading.Event()
        self.playing = threading.Event()
        self.overlay_enabled = threading.Event()
        self.overlay_enabled.set()
        self.latest_preview = deque(maxlen=1)
        self.finished = queue.SimpleQueue()
        self.closing = False

        self.source = tk.StringVar(value="British Highway")
        self.initial = tk.StringVar()
        self.plates = tk.BooleanVar()
        self.email_enabled = tk.BooleanVar()
        self.overlay = tk.BooleanVar(value=True)
        self.status = tk.StringVar()
        self.counts = tk.StringVar(value="Entries 0    Exits 0    Inside 0    0.0 s")
        self.source_name = tk.StringVar()
        self.history_video = tk.StringVar(value="All videos")
        self.history_run = tk.StringVar(value="All runs")
        self.plate_filter = tk.StringVar()
        self.history_status = tk.StringVar()
        self.history_rows = {}
        self.run_controls = []

        root.title("Vehicle Monitor")
        root.geometry("1180x760")
        root.minsize(900, 630)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tabs = ttk.Notebook(root, padding=8)
        self.tabs.pack(fill="both", expand=True)
        self.monitor = ttk.Frame(self.tabs, padding=14)
        self.history = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(self.monitor, text="Monitor")
        self.tabs.add(self.history, text="History")
        self.build_monitor()
        self.build_history()
        self.tabs.bind("<<NotebookTabChanged>>", self.refresh_active_history)
        self.load_video(VIDEO)
        self.refresh_history()
        root.after(33, self.poll_worker)

    def build_monitor(self):
        self.monitor.columnconfigure(0, weight=1)
        self.monitor.rowconfigure(2, weight=1)
        source_bar = ttk.Frame(self.monitor)
        source_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        source = ttk.Combobox(source_bar, textvariable=self.source, state="readonly",
                              values=("British Highway", "Parking"), width=18)
        source.pack(side="left")
        source.bind("<<ComboboxSelected>>", self.select_preset)
        open_video = ttk.Button(source_bar, text="Open video…", command=self.open_video)
        open_video.pack(side="left", padx=8)
        ttk.Label(source_bar, textvariable=self.source_name).pack(side="left", padx=6)
        self.source_picker = source
        self.run_controls.append(open_video)

        options = ttk.Frame(self.monitor)
        options.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(options, text="Starting count").pack(side="left")
        count = ttk.Entry(options, textvariable=self.initial, width=6)
        count.pack(side="left", padx=(6, 4))
        ttk.Label(options, text="Blank = Auto").pack(side="left", padx=(0, 12))
        plates = ttk.Checkbutton(options, text="Read plates", variable=self.plates)
        email = ttk.Checkbutton(options, text="Email alerts", variable=self.email_enabled)
        plates.pack(side="left", padx=(0, 8))
        email.pack(side="left", padx=(0, 8))
        ttk.Checkbutton(options, text="Show IDs and arrows", variable=self.overlay,
                        command=self.toggle_overlay).pack(side="left")
        self.run_controls.extend((count, plates, email))

        self.canvas = tk.Canvas(self.monitor, background="#182129", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.redraw_preview)
        self.canvas.bind("<Button-1>", self.add_zone_point)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        controls = ttk.Frame(self.monitor)
        controls.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.pause_button = ttk.Button(controls, text="Pause", command=self.pause)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop)
        self.draw_button = ttk.Button(controls, text="Draw zone", command=self.begin_zone)
        self.confirm_button = ttk.Button(controls, text="Confirm zone", command=self.confirm_zone)
        self.reset_button = ttk.Button(controls, text="Clear points", command=self.clear_points)
        for button in (self.start_button, self.pause_button, self.stop_button,
                       self.draw_button, self.confirm_button, self.reset_button):
            button.pack(side="left", padx=(0, 5))
        ttk.Label(controls, textvariable=self.counts).pack(side="right")
        ttk.Label(self.monitor, textvariable=self.status, wraplength=840).grid(
            row=4, column=0, sticky="w", pady=(8, 0))
        self.update_controls()

    def build_history(self):
        self.history.columnconfigure(0, weight=1)
        self.history.rowconfigure(1, weight=1)
        filters = ttk.Frame(self.history)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(filters, text="Video").pack(side="left")
        self.video_filter = ttk.Combobox(filters, textvariable=self.history_video,
                                         state="readonly", width=24)
        self.video_filter.pack(side="left", padx=(5, 14))
        ttk.Label(filters, text="Run").pack(side="left")
        self.run_filter = ttk.Combobox(filters, textvariable=self.history_run,
                                       state="readonly", width=24)
        self.run_filter.pack(side="left", padx=(5, 14))
        ttk.Label(filters, text="Plate").pack(side="left")
        search = ttk.Entry(filters, textvariable=self.plate_filter, width=14)
        search.pack(side="left", padx=5)
        ttk.Button(filters, text="Refresh", command=self.refresh_history).pack(side="left", padx=4)
        self.video_filter.bind("<<ComboboxSelected>>", self.change_history_video)
        self.run_filter.bind("<<ComboboxSelected>>", self.refresh_history)
        search.bind("<Return>", self.refresh_history)

        columns = (
            ("time", "Recorded", 165), ("video", "Video", 160),
            ("seconds", "Video time", 85), ("vehicle", "ID", 50),
            ("type", "Vehicle", 80), ("direction", "Crossing", 75),
            ("plate", "Plate", 95), ("email", "Email", 130),
        )
        names = [name for name, _, _ in columns]
        table = ttk.Frame(self.history)
        table.grid(row=1, column=0, sticky="nsew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.events = ttk.Treeview(table, columns=names, show="headings", selectmode="browse")
        for name, label, width in columns:
            self.events.heading(name, text=label)
            self.events.column(name, width=width, minwidth=44, stretch=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.events.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.events.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.events.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        self.events.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.events.bind("<Double-1>", self.open_screenshot)
        self.events.bind("<<TreeviewSelect>>", self.history_selection)

        footer = ttk.Frame(self.history)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.screenshot_button = ttk.Button(footer, text="Open screenshot",
                                            command=self.open_screenshot, state="disabled")
        self.screenshot_button.pack(side="left", padx=(0, 12))
        ttk.Label(footer, textvariable=self.history_status).pack(side="left")

    def select_preset(self, _event=None):
        self.load_video(PARKING_VIDEO if self.source.get() == "Parking" else VIDEO)

    def open_video(self):
        path = filedialog.askopenfilename(parent=self.root, initialdir=VIDEO.parent,
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                                                     ("All files", "*.*")])
        if path:
            self.load_video(Path(path))

    def load_video(self, path):
        capture = cv2.VideoCapture(str(path))
        try:
            frame, _ = read_first_frame(capture, path)
        except (ValueError, cv2.error) as exc:
            self.source.set(self.preset_name())
            self.status.set(str(exc))
            return
        finally:
            capture.release()
        self.video = path
        self.first_frame = self.preview_frame = frame
        self.points.clear()
        self.drawing = False
        name = self.preset_name()
        self.source.set(name)
        self.source_name.set(path.name)
        self.plates.set(name == "Parking")
        self.initial.set("")
        self.counts.set("Entries 0    Exits 0    Inside 0    0.0 s")
        self.zone = None
        if name:
            corners = PARKING_ZONE if name == "Parking" else DEFAULT_ZONE
            self.zone = preset_zone(frame, corners)
        self.status.set("Preset zone ready." if self.zone is not None
                        else "Draw a zone on a fixed-camera clip before starting.")
        self.update_controls()
        self.redraw_preview()

    def preset_name(self):
        if self.video == PARKING_VIDEO:
            return "Parking"
        if self.video == VIDEO:
            return "British Highway"
        return ""

    def begin_zone(self):
        if self.drawing:
            self.drawing = False
            self.status.set("Previous zone restored." if self.zone is not None else "Draw a zone before starting.")
            self.update_controls()
            self.redraw_preview()
            return
        self.preview_frame = self.first_frame
        self.points.clear()
        self.drawing = True
        self.status.set("Click the corners in order, then select Confirm zone.")
        self.update_controls()
        self.redraw_preview()

    def add_zone_point(self, event):
        if not self.drawing or self.image_bounds is None:
            return
        left, top, width, height = self.image_bounds
        if not (left <= event.x < left + width and top <= event.y < top + height):
            return
        frame_height, frame_width = self.first_frame.shape[:2]
        x = int((event.x - left) * frame_width / width)
        y = int((event.y - top) * frame_height / height)
        self.points.append((x, y))
        self.redraw_preview()

    def clear_points(self):
        self.points.clear()
        self.redraw_preview()

    def confirm_zone(self):
        polygon = np.array(self.points, dtype=np.int32)
        if len(set(self.points)) < 3 or cv2.contourArea(polygon) < 100:
            self.status.set("Pick at least three corners that enclose an area.")
            return
        self.zone = polygon
        self.drawing = False
        self.status.set("Custom zone ready.")
        self.update_controls()
        self.redraw_preview()

    def redraw_preview(self, _event=None):
        if self.preview_frame is None:
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if min(canvas_width, canvas_height) < 2:
            return
        frame = self.preview_frame
        height, width = frame.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        left = (canvas_width - size[0]) // 2
        top = (canvas_height - size[1]) // 2
        self.image_bounds = left, top, *size
        image = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas.coords(self.image_item, left, top)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.delete("zone")
        points = self.points if self.drawing else self.zone
        if frame is not self.first_frame or points is None or len(points) == 0:
            return
        screen = [(left + x * size[0] / width, top + y * size[1] / height) for x, y in points]
        if len(screen) >= 2:
            path = screen + screen[:1] if len(screen) >= 3 else screen
            self.canvas.create_line(path, fill="#ffd34e", width=3, tags="zone")
        if self.drawing:
            for x, y in screen:
                self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="#ffd34e", outline="", tags="zone")

    def update_controls(self):
        active = self.worker is not None
        for control in self.run_controls:
            control.configure(state="disabled" if active else "normal")
        self.source_picker.configure(state="disabled" if active else "readonly")
        ready = not active and not self.drawing and self.zone is not None
        self.start_button.configure(state="normal" if ready else "disabled")
        editing = not active and self.drawing
        self.draw_button.configure(state="normal" if not active and self.video else "disabled",
                                   text="Cancel drawing" if self.drawing else "Draw zone")
        self.confirm_button.configure(state="normal" if editing else "disabled")
        self.reset_button.configure(state="normal" if editing else "disabled")
        can_stop = active and not self.stop_requested.is_set()
        self.pause_button.configure(state="normal" if can_stop else "disabled",
                                    text="Pause" if self.playing.is_set() or not active else "Resume")
        self.stop_button.configure(state="normal" if can_stop else "disabled")

    def toggle_overlay(self):
        if self.overlay.get():
            self.overlay_enabled.set()
        else:
            self.overlay_enabled.clear()

    def ask_email(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Email alerts")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        form = ttk.Frame(dialog, padding=16)
        form.pack(fill="both", expand=True)
        fields = []
        for row, label in enumerate(("Sender Gmail address", "Recipient address", "Gmail app password")):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
            entry = ttk.Entry(form, width=35, show="*" if row == 2 else "")
            entry.grid(row=row, column=1, pady=5)
            fields.append(entry)
        settings = None

        def accept(_event=None):
            nonlocal settings
            sender, recipient, password = (field.get().strip() for field in fields)
            if "@" not in sender or "@" not in recipient or not password:
                messagebox.showerror("Email alerts", "Enter both addresses and the app password.",
                                     parent=dialog)
                return
            settings = EmailSettings(sender, recipient, password.replace(" ", ""))
            dialog.destroy()

        ttk.Button(form, text="Start run", command=accept).grid(row=3, column=1, sticky="e", pady=6)
        ttk.Button(form, text="Cancel", command=dialog.destroy).grid(row=3, column=0, sticky="w")
        dialog.bind("<Return>", accept)
        dialog.bind("<Escape>", lambda _: dialog.destroy())
        dialog.wait_visibility()
        dialog.grab_set()
        fields[0].focus_set()
        self.root.wait_window(dialog)
        return settings

    def start(self):
        if self.worker is not None or self.zone is None or self.drawing:
            return
        value = self.initial.get().strip()
        try:
            initial = int(value) if value else None
            if initial is not None and initial < 0:
                raise ValueError
        except ValueError:
            self.status.set("Enter a nonnegative whole number or leave the starting count blank.")
            return
        email = None
        if self.email_enabled.get():
            email = self.ask_email()
            if email is None:
                return
        options = MonitorOptions(self.video, self.weights, self.device, initial, self.plates.get(),
                                 email=email)
        self.latest_preview.clear()
        self.stop_requested.clear()
        self.playing.set()
        self.counts.set("Entries 0    Exits 0    Inside 0    0.0 s")
        self.status.set("Loading models…")
        self.worker = threading.Thread(target=self.run_video, args=(options, self.zone.copy()))
        self.worker.start()
        self.update_controls()

    def run_video(self, options, zone):
        message = "Video finished."
        capture = None
        try:
            if not self.stop_requested.is_set():
                capture = cv2.VideoCapture(str(options.video))
                frame, fps = read_first_frame(capture, options.video)
                with closing(process_video(capture, frame, fps, zone, options,
                                           self.overlay_enabled.is_set)) as stream:
                    while not self.stop_requested.is_set():
                        if not self.playing.wait(.1) or self.stop_requested.is_set():
                            continue
                        started = time.perf_counter()
                        preview = next(stream, None)
                        if preview is None:
                            break
                        self.latest_preview.append(preview)
                        self.stop_requested.wait(max(0, 1 / fps - (time.perf_counter() - started)))
            if self.stop_requested.is_set():
                message = "Stopped."
        except Exception as exc:
            message = f"Could not run monitor. {exc}"
        finally:
            if capture is not None:
                capture.release()
            self.finished.put(message)

    def pause(self):
        if self.playing.is_set():
            self.playing.clear()
            self.status.set("Paused.")
        else:
            self.playing.set()
            self.status.set("Playing.")
        self.update_controls()

    def stop(self):
        self.stop_requested.set()
        self.playing.set()
        self.status.set("Stopping. Finishing any queued email…")
        self.update_controls()

    def close(self):
        if self.worker is None:
            self.root.destroy()
            return
        self.closing = True
        self.stop()

    def poll_worker(self):
        if self.latest_preview:
            preview = self.latest_preview.popleft()
            self.preview_frame = preview.frame
            self.counts.set(f"Entries {preview.entries}    Exits {preview.exits}    "
                            f"Inside {preview.inside}    {preview.video_seconds:.1f} s")
            if self.playing.is_set() and not self.stop_requested.is_set():
                self.status.set(f"Playing. {preview.visible} visible inside." +
                                (" Count differs from view." if preview.count_mismatch else ""))
            self.redraw_preview()
        try:
            message = self.finished.get_nowait()
        except queue.Empty:
            pass
        else:
            self.worker = None
            self.playing.clear()
            if self.closing:
                self.root.destroy()
                return
            self.status.set(message)
            self.update_controls()
            self.refresh_history()
        self.root.after(33, self.poll_worker)

    def refresh_active_history(self, _event=None):
        if self.tabs.select() == str(self.history):
            self.refresh_history()

    def change_history_video(self, _event=None):
        self.history_run.set("All runs")
        self.refresh_history()

    def refresh_history(self, _event=None):
        try:
            uri = DATABASE.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=.25)) as db:
                videos = [row[0] for row in db.execute("SELECT DISTINCT video FROM crossing_log ORDER BY video")]
                video_options = {Path(path).name: path for path in videos}
                if len(video_options) != len(videos):
                    video_options = {path: path for path in videos}
                self.video_filter.configure(values=["All videos", *video_options])
                video = video_options.get(self.history_video.get())
                runs = db.execute("""
                    SELECT DISTINCT session_id FROM crossing_log
                    WHERE (? IS NULL OR video = ?) ORDER BY session_id DESC
                """, (video, video)).fetchall()
                run_options = [row[0] or "Legacy runs" for row in runs]
                self.run_filter.configure(values=["All runs", *run_options])
                run = self.history_run.get()
                if run not in run_options:
                    run = "All runs"
                    self.history_run.set(run)
                run_value = "" if run == "Legacy runs" else run
                rows = db.execute("""
                    SELECT event_id, timestamp, video, video_seconds, vehicle_id, vehicle_type,
                           direction, license_plate, email_status, email_error, image_path
                    FROM crossing_log
                    WHERE (? IS NULL OR video = ?)
                      AND (? = 'All runs' OR COALESCE(session_id, '') = ?)
                      AND INSTR(UPPER(COALESCE(license_plate, '')), ?) > 0
                    ORDER BY event_id DESC LIMIT 500
                """, (video, video, run, run_value, self.plate_filter.get().strip().upper())).fetchall()
        except sqlite3.Error as exc:
            self.history_status.set(f"Could not read event history. {exc}")
            return
        selected = self.events.selection()
        self.events.delete(*self.events.get_children())
        self.history_rows.clear()
        statuses = {"smtp_accepted": "Accepted by Gmail", "failed": "Failed",
                    "disabled": "Off", "not_applicable": "—", "queued": "Queued"}
        for event_id, recorded, video, seconds, vehicle, kind, direction, plate, mail, error, path in rows:
            key = str(event_id)
            self.events.insert("", "end", iid=key,
                               values=(recorded[:19].replace("T", " "), Path(video).name,
                                       f"{seconds:.2f} s", vehicle, kind, direction.title(),
                                       plate or "—", statuses.get(mail, mail)))
            self.history_rows[key] = path, error
        if selected and selected[0] in self.history_rows:
            self.events.selection_set(selected[0])
        self.history_selection()

    def history_selection(self, _event=None):
        selected = self.events.selection()
        path, error = self.history_rows.get(selected[0], (None, None)) if selected else (None, None)
        self.screenshot_button.configure(state="normal" if path else "disabled")
        if error:
            self.history_status.set(f"Email failed. {error}")
        elif self.history_rows:
            self.history_status.set(f"{len(self.history_rows)} recent crossings")
        else:
            self.history_status.set("No crossings match these filters.")

    def open_screenshot(self, _event=None):
        selected = self.events.selection()
        if not selected:
            return
        path, _ = self.history_rows.get(selected[0], (None, None))
        if path:
            try:
                os.startfile(Path(path))
            except OSError as exc:
                messagebox.showerror("Screenshot", f"Could not open the image. {exc}", parent=self.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--device")
    args = parser.parse_args()
    tcl_root = Path(sys.base_prefix) / "tcl"
    for key, folder in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        if (tcl_root / folder).is_dir():
            os.environ.setdefault(key, str(tcl_root / folder))
    root = tk.Tk()
    try:
        initialize_database(DATABASE)
    except sqlite3.Error as exc:
        messagebox.showerror("Vehicle Monitor", f"Could not open the database. {exc}", parent=root)
        root.destroy()
        return
    VehicleApp(root, args.weights, args.device)
    root.mainloop()


if __name__ == "__main__":
    main()
