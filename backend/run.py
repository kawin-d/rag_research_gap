import sys
import uvicorn
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings

if __name__ == "__main__":
    host = settings.HOST
    port = settings.PORT
    print("=========================================================================")
    print(" Starting RAG Research Paper Analysis & Gap Identification Backend Server")
    print(f" Web UI available at:            http://{host}:{port}/ui")
    print(f" API Documentation available at: http://{host}:{port}/docs")
    print(f" OpenAPI Specification at:       http://{host}:{port}/openapi.json")
    print("=========================================================================")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
