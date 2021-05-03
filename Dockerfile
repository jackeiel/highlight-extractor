FROM python:3.7
MAINTAINER Jack Eiel "jackneiel@gmail.com"
COPY . .
WORKDIR /app
RUN apt-get -y update
RUN pip install -upgrade pip
RUN pip install numpy pandas tensorflow keras-bert spacy nltk
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt')"
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]