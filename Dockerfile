FROM ubuntu:latest
RUN apt update
RUN apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    libfftw3-dev \
    git-lfs \
    wget \
    curl \
    vim
RUN python -m venv weather
RUN bash /weather/bin/activate
RUN pip install pandas numpy scikit-learn seaborn matplotlib torch
RUN find -name "*.pyc" -exec rm {} \;
RUN git clone https://github.com/yashpatel5400/stats604-project4.git
WORKDIR /stats604-project4