import pytz
from fastapi import FastAPI, Request
from datetime import datetime

app = FastAPI(
    title="ViT Implementation",
    description="ViT Implementation APIs and demo page for the paper: https://arxiv.org/pdf/2010.11929.pdf",
    version="1.0.0"
)

IST = pytz.timezone('Asia/Kolkata')
started_at = datetime.now(IST)


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
