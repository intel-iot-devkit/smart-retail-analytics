FROM ubuntu:16.04

ADD . /app
WORKDIR /app
ENV http_proxy 'http://proxy-iind.intel.com:911'
ENV https_proxy 'http://proxy-iind.intel.com:911'

ARG INSTALL_DIR=/opt/intel/openvino
ARG OPENVINO=l_openvino_toolkit_p_2020.3.194.tgz
ARG COUNTRY=1
ARG IMAGE=mssi
ARG VERSION=1.0

RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Install needed dependences
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-setuptools \
        wget \
        sudo

#Upgrading the pip version
RUN pip3 install --upgrade pip

#Downloading Openvino toolkit
RUN cd /app && \
    wget -O $OPENVINO http://registrationcenter-download.intel.com/akdlm/irc_nas/16670/$OPENVINO

RUN tar -xvzf $OPENVINO

# Installing OpenVINO dependencies
RUN cd /app/l_openvino_toolkit* && \
   ./install_openvino_dependencies.sh

RUN pip3 install numpy requests influxdb flask pyyaml pathlib

# Installing OpenVINO itself
RUN cd /app/l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"

RUN echo "source $INSTALL_DIR/bin/setupvars.sh" >> /root/.bashrc

RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites/ && \
    ./install_prerequisites.sh && \
    ./install_prerequisites_caffe.sh

RUN cd $INSTALL_DIR/deployment_tools/tools/model_downloader && \
    ./downloader.py --name face-detection-adas-0001 && \
    ./downloader.py --name head-pose-estimation-adas-0001 && \
    ./downloader.py --name emotions-recognition-retail-0003 && \
    ./downloader.py --name person-detection-retail-0002 && \
    ./downloader.py --name mobilenet-ssd

RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/ && \
    ./mo_caffe.py --input_model $INSTALL_DIR/deployment_tools/open_model_zoo/tools/downloader/public/mobilenet-ssd/mobilenet-ssd.caffemodel  -o /app/resources/FP32 --data_type FP32 --scale 256 --mean_values [127,127,127] && \
    ./mo_caffe.py --input_model $INSTALL_DIR/deployment_tools/open_model_zoo/tools/downloader/public/mobilenet-ssd/mobilenet-ssd.caffemodel  -o /app/resources/FP16 --data_type FP16 --scale 256 --mean_values [127,127,127]

RUN cd /app/resources && \
    wget -O face-demographics-walking.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4 && \
    wget -O bottle-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4 && \
    wget -O head-pose-face-detection-female.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4

CMD ["/bin/bash"]
