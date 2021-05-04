FROM python:3.7
MAINTAINER Jack Eiel "jackneiel@gmail.com"
COPY . .
WORKDIR .
RUN apt-get -y update
RUN pip install --upgrade pip
RUN pip install numpy pandas scikit-learn tensorflow keras-bert spacy nltk flask
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt')"
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]