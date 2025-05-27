FROM python:3.10

RUN useradd -m biopb
ENV HOME=/home/biopb

USER biopb
WORKDIR $HOME

ENV PATH="$PATH:$HOME/.local/bin/"

RUN wget https://huggingface.co/jiyuuchc/lacss3-base/resolve/main/lacss3-base?download=true -O lacss3-base

RUN mkdir jax_cache

# copy rest of the code
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ARG VERSION=0.15.2
RUN pip install --no-cache-dir --no-deps lacss==$VERSION

EXPOSE 50051

ENV JAX_COMPILATION_CACHE_DIR="./jax_cache"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

ENTRYPOINT ["python", "-m", "lacss.deploy.remote_server"]

CMD ["--no-token", "--f16", "--modelpath", "./lacss3-base"]
