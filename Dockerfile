FROM python:3.11.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# System dependency used by numerical libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep Poetry separate from the project environment.
RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install "poetry==2.4.0"

# Install locked application dependencies before copying source code.
COPY pyproject.toml poetry.lock README.md ./

RUN /opt/poetry/bin/poetry install --only main --no-root

# Install the application package.
COPY src/ ./src/
COPY params.yaml ./params.yaml

RUN /opt/poetry/bin/poetry install --only-root

CMD ["python", "-m", "purchase_propensity.modeling.train", "--help"]
