import os

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import torch

from src.db.database import engine, Base
from src.api.router import router
from src.db.models import InspectionLog
from src.model import DefectClassifier

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("Loading PyTorch model...")
    model = DefectClassifier()
    weights_path = os.path.join(base_dir, 'models', 'defect_model_weights.pth')
    weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    app.state.model = model
    yield
    print("Model unloaded")

app = FastAPI(title="Wafer Defect API", lifespan=lifespan)
media_path = os.path.join(base_dir, "media")
app.mount("/media", StaticFiles(directory=media_path), name="media")
app.include_router(router)