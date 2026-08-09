import sqlalchemy
from datetime import datetime
from src.db.database import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer,Float,DateTime

class InspectionLog(Base):
    __tablename__ = "inspection_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_name: Mapped[str] = mapped_column(String)
    predicted_class: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime,server_default=sqlalchemy.sql.func.now())
