from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, model, predict, feedback

api_router = APIRouter()

api_router.include_router(health.router, prefix="", tags=["Health"])
api_router.include_router(model.router, prefix="/model", tags=["Model"])
api_router.include_router(predict.router, prefix="/predict", tags=["Prediction"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
