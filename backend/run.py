import sys
import uvicorn
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    print("=========================================================================")
    print(" Starting RAG Research Paper Analysis & Gap Identification Backend Server")
    print(" API Documentation available at: http://127.0.0.1:8000/docs")
    print(" OpenAPI Specification at:       http://127.0.0.1:8000/openapi.json")
    print("=========================================================================")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
