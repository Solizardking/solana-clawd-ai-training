"""Hugging Face Space: chart API plus authenticated Spark model endpoint."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

os.environ.setdefault('CHART_MODEL_BACKEND', 'spark')
os.environ.setdefault('CHART_MODEL_NAME', 'clawd-spark')
os.environ.setdefault('LLAMA_URL', 'http://127.0.0.1:7860/model')
from . import server, spark_server


@asynccontextmanager
async def lifespan(app):
    async with spark_server.lifespan(spark_server.app):
        async with server.lifespan(server.app):
            yield


app = FastAPI(lifespan=lifespan)
app.mount('/model', spark_server.app)
app.mount('/', server.app)
