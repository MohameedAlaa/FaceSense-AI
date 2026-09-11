from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.api import api_router
from backend.app.core.config import settings
from backend.app.schemas.common import HealthResponse
from backend.app.services.prediction_service import (
    ImageDecodeError,
    NoFaceDetectedError,
    BackendError,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Exception Handlers
@app.exception_handler(ImageDecodeError)
async def image_decode_exception_handler(request: Request, exc: ImageDecodeError):
    return JSONResponse(
        status_code=400,
        content={"error": "ImageDecodeError", "detail": str(exc)},
    )


@app.exception_handler(NoFaceDetectedError)
async def no_face_exception_handler(request: Request, exc: NoFaceDetectedError):
    return JSONResponse(
        status_code=422,
        content={"error": "NoFaceDetectedError", "detail": str(exc)},
    )


@app.exception_handler(BackendError)
async def generic_backend_exception_handler(request: Request, exc: BackendError):
    return JSONResponse(
        status_code=500,
        content={"error": "BackendError", "detail": str(exc)},
    )


# Root Health Check
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def root_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
    )


# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)
