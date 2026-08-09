from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,DeclarativeBase
from typing import Generator
import os
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/defects_db"
)
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo = True)

SessionLocal = sessionmaker(bind = engine,autocommit = False, autoflush = False)

class Base(DeclarativeBase):
    pass

def get_db() -> Generator:
    db = SessionLocal()

    try:
        yield db

    finally: db.close()