FROM python:alpine3.7
COPY . /app
WORKDIR /app
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]