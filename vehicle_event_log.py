"""Persist crossings and send optional Gmail alerts outside the video loop."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
import os
from pathlib import Path
import smtplib
import sqlite3
import ssl


@dataclass(frozen=True)
class EmailSettings:
    sender: str
    recipient: str
    password: str = field(repr=False)

    @classmethod
    def from_environment(cls):
        values = [os.getenv("VEHICLE_ALERT_" + name)
                  for name in ("SENDER", "RECIPIENT", "APP_PASSWORD")]
        if all(values):
            return cls(*values)
        return None


def initialize_schema(connection):
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                video TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                event TEXT NOT NULL,
                video_seconds REAL NOT NULL,
                frame INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                vehicle_class TEXT NOT NULL,
                estimated_inside INTEGER NOT NULL,
                count_mismatch INTEGER NOT NULL,
                screenshot TEXT,
                basis TEXT NOT NULL,
                visible_inside INTEGER NOT NULL,
                starting_inside INTEGER NOT NULL,
                starting_count_mode TEXT NOT NULL,
                email_status TEXT NOT NULL,
                email_error TEXT
            )
        """)
        columns = connection.execute("PRAGMA table_info(events)").fetchall()
        if "session_id" not in {column[1] for column in columns}:
            connection.execute("ALTER TABLE events ADD COLUMN session_id TEXT")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_plates (
                session_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                plate TEXT NOT NULL,
                PRIMARY KEY (session_id, track_id)
            )
        """)
        view = connection.execute("PRAGMA table_info(crossing_log)").fetchall()
        if {column[1] for column in view} >= {"license_plate", "session_id", "video", "event_id"}:
            return
        connection.execute("DROP VIEW IF EXISTS crossing_log")
        connection.execute("""
            CREATE VIEW crossing_log AS
            SELECT e.recorded_at AS timestamp, e.video_seconds, e.track_id AS vehicle_id,
                   e.vehicle_class AS vehicle_type, e.event AS direction,
                   CASE WHEN e.screenshot <> '' THEN e.run_dir || '/' || e.screenshot END AS image_path,
                   e.email_status, p.plate AS license_plate,
                   e.id AS event_id, e.video, e.session_id, e.email_error
            FROM events e LEFT JOIN vehicle_plates p
                ON e.session_id = p.session_id AND e.track_id = p.track_id
        """)


def initialize_database(path):
    with closing(sqlite3.connect(path)) as connection:
        initialize_schema(connection)


class EventLog:
    def __init__(self, path, video, run_dir, email=None):
        self.path = path
        self.video = str(video.resolve())
        self.run_dir = str(run_dir.resolve())
        self.email = email
        self.enabled = email is not None
        self.connection = sqlite3.connect(path)
        try:
            initialize_schema(self.connection)
        except Exception:
            self.connection.close()
            raise
        self.worker = ThreadPoolExecutor(max_workers=1) if self.enabled else None
        print("Email alerts enabled." if self.enabled else "Email alerts disabled.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.worker:
                self.worker.shutdown(wait=True)
        finally:
            self.connection.close()
            self.email = None

    def record_plate(self, session_id, track_id, plate):
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO vehicle_plates VALUES (?, ?, ?)",
                                    (session_id, track_id, plate))

    def record(self, row, session_id=None):
        entry = row[0] == "entry"
        status = ("queued" if self.enabled else "disabled") if entry else "not_applicable"
        recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.connection:
            event_id = self.connection.execute("""
                INSERT INTO events (
                    recorded_at, video, run_dir, event, video_seconds, frame, track_id,
                    vehicle_class, estimated_inside, count_mismatch, screenshot, basis,
                    visible_inside, starting_inside, starting_count_mode, email_status, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (recorded_at, self.video, self.run_dir, *row, status, session_id)).lastrowid
        if status == "queued":
            self.worker.submit(self.send, event_id, recorded_at, row)

    def send(self, event_id, recorded_at, row):
        _, seconds, _, track_id, vehicle_class, inside, _, screenshot, *_ = row
        error = None
        stage = "preparing message"
        try:
            msg = EmailMessage()
            msg["From"] = self.email.sender
            msg["To"] = self.email.recipient
            msg["Subject"] = f"Vehicle entered: {vehicle_class} #{track_id}"
            msg.set_content(
                f"Recorded at {recorded_at}\n"
                f"Video: {Path(self.video).name}\n"
                f"Playback time: {seconds:.2f} seconds\n"
                f"Vehicle: {vehicle_class} #{track_id}\n"
                f"Estimated inside: {inside}\n"
            )
            if screenshot:
                msg.add_attachment(
                    (Path(self.run_dir) / screenshot).read_bytes(),
                    maintype="image", subtype="jpeg", filename=screenshot,
                )
            stage = "connecting to Gmail"
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10,
                                  context=ssl.create_default_context()) as smtp:
                stage = "signing in to Gmail"
                smtp.login(self.email.sender, self.email.password)
                stage = "sending message"
                smtp.send_message(msg)
                stage = "closing Gmail connection"
            status = "smtp_accepted"
        except (OSError, ValueError, smtplib.SMTPException) as exc:
            status = "failed"
            detail = " ".join(str(exc).split())
            if self.email.password:
                detail = detail.replace(self.email.password, "[redacted]")
            error = f"{type(exc).__name__} during {stage}: {detail}"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "UPDATE events SET email_status = ?, email_error = ? WHERE id = ?",
                (status, error, event_id),
            )
        print(f"Email for event {event_id}: {status}" + (f" ({error})" if error else ""))
