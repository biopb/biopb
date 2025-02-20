FROM python:3.8
RUN useradd -m biopb
ENV HOME=/home/biopb
USER biopb
WORKDIR $HOME

ENV PATH="$PATH:$HOME/.local/bin/"

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# pull model data
RUN python -c 'from cellpose import models; models.Cellpose("cyto3")'


# copy rest of the code
COPY . .

EXPOSE 50051

ENTRYPOINT ["python", "./cellpose_server.py"]

CMD ["--no-token"]
