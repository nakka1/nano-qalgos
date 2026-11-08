# docker build -t nano-qalgos . && docker run --rm -it nano-qalgos
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md nanoqalgos.py nanoqalgos_cli.py nanoqalgos_math.py ./
RUN pip install . && useradd -m quantum
USER quantum
ENTRYPOINT ["nanoqalgos"]
CMD ["demo"]
