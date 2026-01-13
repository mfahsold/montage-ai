from fastapi import FastAPI, Depends, HTTPException
from typing import Optional


app = FastAPI()


@app.get("/health")
async def health(status: Optional[str] = None):
    return {"status": "ok", "detail": status}
