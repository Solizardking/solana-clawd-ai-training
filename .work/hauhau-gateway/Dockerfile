# Pinned CPU llama.cpp; web gateway stays available while the GGUF loads.
FROM ghcr.io/ggml-org/llama.cpp:server@sha256:8459c3e6a16f4e1d92fecd0ad8d015169e9f108c1862a366eb44923192b576f8
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --chmod=755 entrypoint.sh /entrypoint.sh
COPY gateway.py /gateway.py
COPY www /www
ENV LLAMA_CACHE=/models PORT=8080 HAUHAU_MODEL_PROFILE=qwen36 LLAMA_CTX=8192 LLAMA_THREADS=4 HAUHAU_WWW=/www
EXPOSE 8080
ENTRYPOINT ["python3", "/gateway.py"]
