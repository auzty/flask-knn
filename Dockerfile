FROM python:3.7-slim-buster

RUN useradd -m -s /bin/bash --uid 1000 python 

WORKDIR /opt/apps

ADD . /opt/apps

RUN chown python:python -R /opt/apps

USER python

RUN pip install -r requirements.txt

# Train and Generate Model
RUN python train.py

CMD ["python","mainapp.py"]