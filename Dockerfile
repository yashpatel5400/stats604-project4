FROM ubuntu:latest
RUN apt update
RUN apt install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    libfftw3-dev \
    git-lfs \
    wget \
    curl \
    vim
RUN python -m pip install poetry
COPY pyproject.toml poetry.lock ./
RUN mkdir ./predictor && touch ./predictor/__init__.py
RUN poetry install --no-interaction --ansi
COPY predictor ./predictor/
RUN find -name "*.pyc" -exec rm {} \;
CMD ["bash"]