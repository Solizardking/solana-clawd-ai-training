FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS llama-build
RUN apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build build-essential libcurl4-openssl-dev libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
ARG LLAMA_CPP_REV=7992aa7c8e21ea2eb7a5e4802da56eec7b376036
RUN git init /src && cd /src && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch --depth 1 origin "$LLAMA_CPP_REV" && git checkout FETCH_HEAD
RUN cmake -S /src -B /src/build -G Ninja -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" -DCMAKE_BUILD_TYPE=Release && cmake --build /src/build --target llama-server -j 4

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS model
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libcurl4t64 libssl3t64 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=llama-build /src/build/bin/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama
USER 10001:10001
ENTRYPOINT ["/opt/llama/llama-server"]

FROM python:3.13-slim AS api
WORKDIR /app
COPY chart_agent/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY chart_agent/ /app/chart_agent/
COPY docs/solgpt-chart-tool-catalog.md /app/docs/solgpt-chart-tool-catalog.md
USER 10001:10001
CMD ["python", "-m", "uvicorn", "chart_agent.server:app", "--host", "0.0.0.0", "--port", "8090", "--no-access-log"]
