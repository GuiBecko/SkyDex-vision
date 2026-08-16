FROM python:3.11-slim

WORKDIR /srv

# Single constrained install. open_clip_torch declares `torch>=1.9.0` with no
# upper bound and no index preference, so a plain `pip install -r
# requirements.txt` resolves a CUDA wheel over a CPU one bought moments
# earlier -- which is how a 1.5GB image becomes a 5GB one on a machine that
# has no NVIDIA GPU. constraints.txt pins the CPU builds of torch and
# torchvision so pip's resolver can never quietly swap them, and
# --extra-index-url gives it a place to actually find those CPU wheels.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
        -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the weights into the image. Without this the first request after every
# container start waits on a 350MB download, and a container that cannot reach
# the internet never becomes useful at all.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

COPY app ./app

EXPOSE 8000

# One worker. The model is ~350MB of resident memory per process and this
# machine has 14GB total shared with Postgres and the JVM; a second worker buys
# throughput nobody needs and costs memory that is genuinely scarce.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
