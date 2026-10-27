FROM python:3.11.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/app/.venv/bin:/opt/poetry/bin:$PATH"

WORKDIR /app

# Biblioteca de sistema utilizada por dependências numéricas.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala o Poetry em um ambiente separado.
RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install "poetry==2.4.0"

# Instala as dependências do lock file, incluindo o DVC.
COPY pyproject.toml poetry.lock README.md ./

RUN poetry install --with dev --no-root

# Copia o código e a configuração do pipeline.
COPY src/ ./src/
COPY params.yaml dvc.yaml dvc.lock ./

RUN poetry install --only-root

# Inicializa um ambiente DVC independente dentro da imagem.
RUN dvc init --no-scm

CMD ["python", "-m", "purchase_propensity.modeling.train", "--help"]
