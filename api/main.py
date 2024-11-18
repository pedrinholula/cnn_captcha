from fastapi import FastAPI
from routes import predict, health_check

app = FastAPI()

app.include_router(health_check.router, prefix="/ping", tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])