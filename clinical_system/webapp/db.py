"""Database models (SQLAlchemy 2). Works with SQLite and PostgreSQL unchanged."""
from __future__ import annotations

import json
from datetime import date, datetime

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
                        create_engine, inspect, select, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16), default="doctor")  # doctor | admin
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)  # PAT-2026-0001
    name: Mapped[str] = mapped_column(String(120), index=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str] = mapped_column(String(16), default="unknown")
    contact: Mapped[str] = mapped_column(String(120), default="")
    visit_date: Mapped[date] = mapped_column(Date, default=date.today)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)  # created by seed_demo.py
    created_by: Mapped[User] = relationship()
    scans: Mapped[list["Scan"]] = relationship(back_populates="patient", order_by="Scan.created_at.desc()",
                                               cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    body_site: Mapped[str] = mapped_column(String(40), default="unknown")
    image_path: Mapped[str] = mapped_column(String(300))
    gradcam_path: Mapped[str] = mapped_column(String(300), default="")
    lime_path: Mapped[str] = mapped_column(String(300), default="")
    lime_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|done|failed
    lime_summary: Mapped[str] = mapped_column(Text, default="")
    predicted_class: Mapped[str] = mapped_column(String(16))
    predicted_name: Mapped[str] = mapped_column(String(120))
    risk: Mapped[str] = mapped_column(String(16))            # malignant | premalignant | benign
    confidence: Mapped[float] = mapped_column(Float)
    confidence_level: Mapped[str] = mapped_column(String(8))  # High | Medium | Low
    top3_json: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    meaning: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(Text)
    warning: Mapped[str] = mapped_column(Text, default="")
    model_name: Mapped[str] = mapped_column(String(120))
    # timings in milliseconds (None for scans made before timing was recorded)
    time_predict_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_gradcam_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_lime_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    patient: Mapped[Patient] = relationship(back_populates="scans")
    doctor: Mapped[User] = relationship()
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="scan", cascade="all, delete-orphan",
                                                      order_by="Feedback.updated_at")
    # all class probabilities, lesion measurements and photo quality check (None for older scans)
    extra: Mapped["ScanExtra | None"] = relationship(back_populates="scan", cascade="all, delete-orphan",
                                                     uselist=False, lazy="joined")

    @property
    def probs_json(self) -> str:
        return self.extra.probs_json if self.extra else ""

    @property
    def features_json(self) -> str:
        return self.extra.features_json if self.extra else ""

    @property
    def quality_json(self) -> str:
        return self.extra.quality_json if self.extra else ""

    @property
    def time_total_ms(self) -> float | None:
        """Prediction + Grad-CAM (+ LIME once it has finished)."""
        if self.time_predict_ms is None:
            return None
        return sum(t for t in (self.time_predict_ms, self.time_gradcam_ms, self.time_lime_ms) if t is not None)

    def feedback_by(self, user_id: int) -> "Feedback | None":
        return next((f for f in self.feedback if f.doctor_id == user_id), None)

    @property
    def top3(self) -> list[dict]:
        return json.loads(self.top3_json)

    @property
    def probs(self) -> dict:
        return json.loads(self.probs_json) if self.probs_json else {}

    @property
    def features(self) -> dict:
        return json.loads(self.features_json) if self.features_json else {}

    @property
    def quality(self) -> dict:
        return json.loads(self.quality_json) if self.quality_json else {}

    @property
    def uses_metadata(self) -> bool:
        return "metadata" in (self.model_name or "").lower()

    @property
    def is_disease(self) -> bool:
        return self.risk in ("malignant", "premalignant")


class ScanExtra(Base):
    """Added in v1.5 as its own table (so older databases need no change to the scans table): the prior-
    corrected probabilities of ALL classes (as used for the prediction), the lesion measurements behind the
    "why this result" explanation, and the photo quality check, as JSON."""
    __tablename__ = "scan_extras"
    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), unique=True, index=True)
    probs_json: Mapped[str] = mapped_column(Text, default="")
    features_json: Mapped[str] = mapped_column(Text, default="")
    quality_json: Mapped[str] = mapped_column(Text, default="")
    scan: Mapped[Scan] = relationship(back_populates="extra")


class Feedback(Base):
    """A clinician's assessment of one AI result (one per doctor per scan; re-submitting updates it)."""
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("scan_id", "doctor_id", name="uq_feedback_scan_doctor"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), index=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    agreement: Mapped[str] = mapped_column(String(10))                            # agree | disagree | unsure
    correct_class: Mapped[str | None] = mapped_column(String(16), nullable=True)  # only when disagree
    usefulness: Mapped[int] = mapped_column(Integer)                              # 1-5 stars
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    scan: Mapped[Scan] = relationship(back_populates="feedback")
    doctor: Mapped[User] = relationship()


# columns added after the first release: (table, column, SQL type); added in place to existing databases
_NEW_COLUMNS = [
    ("patients", "is_demo", "BOOLEAN DEFAULT FALSE"),
    ("scans", "time_predict_ms", "FLOAT"),
    ("scans", "time_gradcam_ms", "FLOAT"),
    ("scans", "time_lime_ms", "FLOAT"),
]


def migrate(eng=None) -> list[str]:
    """Adds missing columns to an existing database (SQLite or PostgreSQL). Existing data is kept."""
    eng = eng or engine
    insp = inspect(eng)
    tables = set(insp.get_table_names())
    added = []
    with eng.begin() as con:
        for table, col, sqltype in _NEW_COLUMNS:
            if table in tables and col not in {c["name"] for c in insp.get_columns(table)}:
                con.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}"))
                added.append(f"{table}.{col}")
    return added


def init_db() -> None:
    migrate()
    Base.metadata.create_all(engine)


def next_patient_code(db: Session, year: int | None = None) -> str:
    year = year or date.today().year
    prefix = f"PAT-{year}-"
    codes = db.scalars(select(Patient.patient_code).where(Patient.patient_code.like(prefix + "%"))).all()
    n = max((int(c.rsplit("-", 1)[1]) for c in codes), default=0) + 1
    return f"{prefix}{n:04d}"
