FROM ubuntu:16.04

ADD . /app
WORKDIR /app
ENV http_proxy 'http://proxy-iind.intel.com:911'
ENV https_proxy 'http://proxy-iind.intel.com:911'
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install numpy requests influxdb flask pyyaml pathlib
