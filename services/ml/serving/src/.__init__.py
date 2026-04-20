import uvicorn
from fastapi import FastAPI,

app = FastAPI(
    title=""
    description="",
    version="",
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )