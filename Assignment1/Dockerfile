FROM tensorflow/tensorflow:latest

ENV LANG C.UTF-8
#RUN apt-get update && apt-get install -y \
#    software-properties-common
#RUN add-apt-repository universe
#RUN apt-get update && apt-get install -y \
#    python3-pip\
#    python3.8
#RUN pip install tensorflow tensorflow-gpu
MAINTAINER nitishshukla86@gmail.com 
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt