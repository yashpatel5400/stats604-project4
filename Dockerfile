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
RUN python -m venv weather
RUN source ./weather/bin/activate
RUN pip install pandas numpy scikit-learn seaborn matplotlib torch
RUN mkdir ./predictor && touch ./predictor/__init__.py
COPY predictor ./predictor/
RUN find -name "*.pyc" -exec rm {} \;
CMD ["bash"]