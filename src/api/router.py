import os
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from src.ml.preprocessor import process_image
import torch
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import InspectionResponse
from src.db.models import InspectionLog
from src.db.database import get_db

CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "media", "defects")
os.makedirs(UPLOAD_DIR, exist_ok=True)
router = APIRouter(prefix="/api/v1")
@router.post("/predict", response_model=InspectionResponse, description="""
Upload a defect image for classification.

**Testing samples (right-click -> Save image as):**
* [Defect example 1 (Crazing)](/media/defects/crazing_sample.jpg)
* [Defect example 2 (Scratches)](/media/defects/scratches_sample.jpg)
""")
def predict_defect(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):

    model = request.app.state.model
    try:
        image_bytes = file.file.read()
        tensor = process_image(image_bytes)
        file_extension = file.filename.split(".")[-1]
        safe_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file or format")

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)
        predicted_index_item = predicted_index.item()
        predicted_class = CLASSES[predicted_index_item]

    new_log = InspectionLog(image_name=safe_filename, predicted_class=predicted_class, confidence=confidence.item())
    try:
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        return new_log
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error while saving prediction")