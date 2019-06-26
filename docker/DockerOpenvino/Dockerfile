FROM ubuntu:16.04


ADD . /app
WORKDIR /app

ARG INSTALL_DIR=/opt/intel/openvino

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
        sudo 


# installing OpenVINO dependencies
RUN cd /app/l_openvino_toolkit* && \ 
   ./install_openvino_dependencies.sh

RUN pip3 install numpy requests influxdb flask pyyaml pathlib

# installing OpenVINO itself
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
    ./downloader.py --name face-detection-adas-0001-fp16 && \
    ./downloader.py --name head-pose-estimation-adas-0001-fp16 && \
    ./downloader.py --name emotions-recognition-retail-0003-fp16 &&\
    ./downloader.py --name mobilenet-ssd

RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/ && \
    ./mo_caffe.py --input_model $INSTALL_DIR/deployment_tools/tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  -o /app/resources/FP32 --data_type FP32 --scale 256 --mean_values [127,127,127] && \
    ./mo_caffe.py --input_model $INSTALL_DIR/deployment_tools/tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  -o /app/resources/FP16 --data_type FP16 --scale 256 --mean_values [127,127,127]


CMD ["/bin/bash"]
