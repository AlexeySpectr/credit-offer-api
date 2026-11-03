from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class PredictionRecord(Base):

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    decision_day: Mapped[str] = mapped_column(String(20))

    offered_rate: Mapped[float] = mapped_column(Float)

    cb_rate: Mapped[float] = mapped_column(Float)

    loan_amount_last: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    fl_adminarea: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    prediction: Mapped[int] = mapped_column(Integer)

    probability: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.now
    )