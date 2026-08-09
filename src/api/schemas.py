from pydantic import BaseModel, ConfigDict
from datetime import datetime

class InspectionResponse(BaseModel):
    id: int
    image_name: str
    predicted_class: str
    confidence: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)