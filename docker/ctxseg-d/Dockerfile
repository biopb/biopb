FROM python:3.10
RUN useradd -m biopb
ENV HOME=/home/biopb

USER biopb
WORKDIR $HOME

COPY --chown=bipob requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the code
COPY --chown=biopb --from=base . .
COPY . .

# fix file permission 
RUN chmod -R 777 $HOME

EXPOSE 50051

ENTRYPOINT ["XLA_PYTHON_CLIENT_PREALLOCATE=false", "python", "./server.py"]

CMD ["modelpath model_d.pickle", "--no-token"]
