import pytz
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="ViT Implementation",
    description="ViT Implementation APIs and demo page for the paper: https://arxiv.org/pdf/2010.11929.pdf",
    version="1.0.0"
)

IST = pytz.timezone('Asia/Kolkata')
started_at = datetime.now(IST)


class ImageSchema(BaseModel):
    """
    This is the schema for the image.
    """
    image: str


class PredictSchema(BaseModel):
    """
    This is the schema for the prediction.
    """
    label: str
    match: float


class PredictionResponse(BaseModel):
    """
    This is the schema for the response.
    """
    status: str
    results: Optional[List[PredictSchema]]


@app.get('/')
async def root():
    return {
        "project": "ViT Implementation",
        "status": "OK",
        "time_up": started_at,
        "description": (
            "ViT Implementation APIs and demo page for "
            "the paper: https://arxiv.org/pdf/2010.11929.pdf"
        ),
    }

@app.get('/train')
async def train():
    return { "status": "Under Development"}


@app.post('/train')
async def train(label: str, image: ImageSchema, ):
    return { "status": "Under Development"}


@app.post('/predict', response_model=PredictionResponse, response_model_exclude_unset=True)
async def predict(image: ImageSchema):
    resp = PredictionResponse.construct(
        status="Under Development"
    )
    return resp

