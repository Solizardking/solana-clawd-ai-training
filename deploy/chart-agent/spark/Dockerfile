FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers==4.57.1 accelerate==1.10.1 peft==0.17.1 sentencepiece==0.2.1 fastapi==0.115.12 uvicorn==0.34.2 httpx==0.28.1 onnxruntime==1.22.1 pillow==11.3.0 numpy==2.2.6 websockets==15.0.1 huggingface_hub==0.36.0
WORKDIR /app
COPY chart_agent /app/chart_agent
COPY docs /app/docs
COPY outputs/chart-agent/assets/weights/best.onnx /app/outputs/chart-agent/assets/weights/best.onnx
COPY outputs/chart-agent/assets/chart-pattern.onnx /app/outputs/chart-agent/assets/chart-pattern.onnx
COPY outputs/chart-agent/research.sqlite /app/outputs/chart-agent/research.sqlite
COPY outputs/chart-agent/examples.sqlite /app/outputs/chart-agent/examples.sqlite
RUN useradd -m -u 1000 user && mkdir -p /app/outputs/chart-agent/assets && chown -R user:user /app
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "chart_agent.spark_stack:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
