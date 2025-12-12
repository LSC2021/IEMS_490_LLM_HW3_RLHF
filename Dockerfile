# ---------- Base image with CUDA ----------
# If the grader does not use GPU, this will still work as a normal Ubuntu image.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------- System dependencies ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ---------- Python dependencies ----------
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA (adjust version if needed for the grader)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# Core libraries used across reward model, PPO, GRPO, DPO, and analysis
RUN pip install \
    transformers==4.46.0 \
    datasets \
    accelerate \
    pandas \
    numpy \
    tqdm \
    matplotlib \
    scikit-learn \
    openpyxl \
    sentencepiece \
    einops \
    openai

# Optional: jupyter for running `results_analysis.ipynb` inside the container
RUN pip install jupyter

# ---------- Project setup ----------
WORKDIR /workspace

# Copy everything in the repo into the image
COPY . /workspace

# For deterministic text output & better logging behavior
ENV PYTHONUNBUFFERED=1

# HuggingFace cache inside the container (prevents cluttering root)
ENV HF_HOME=/workspace/.cache/huggingface

# By default drop into a shell; the grader can choose which script to run.
# Example inside container:
#   python data_process.py
#   python reward_model_training.py
#   python PPO_training.py
#   python GRPO_training.py
#   python DPO_training.py
#   python testing.py
CMD ["bash"]
