FROM python:3.12-slim
RUN useradd -m -u 1000 clawd
WORKDIR /app
COPY --chown=clawd:clawd app.py index.html ./
USER clawd
EXPOSE 7860
CMD ["python", "app.py"]
