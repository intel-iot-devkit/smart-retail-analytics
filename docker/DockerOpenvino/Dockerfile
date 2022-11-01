
FROM openvino/ubuntu20_dev:2022.2.0

ENV http_proxy 'http://proxy-dmz.intel.com:911'
ENV https_proxy 'http://proxy-dmz.intel.com:912'
ENV no_proxy '127.0.0.1,localhost,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.svc.cluster.local,intel.com,10.2.127.101,.intel.com'

USER root
RUN omz_downloader --name face-detection-adas-0001,head-pose-estimation-adas-0001,emotions-recognition-retail-0003,person-detection-retail-0002,mobilenet-ssd

RUN mkdir -p /app/resources

RUN apt-get update && apt-get upgrade -y && apt-get install -y wget && apt-get clean
RUN pip3 install openvino-dev[caffe]==2022.2.0
RUN pip3 install protobuf==3.20.3
RUN pip3 install influxdb==5.3.1
RUN pip3 install flask==2.2.2

RUN apt-get install -y libopencv-dev python3-opencv

RUN mo --input_model /opt/intel/openvino_2022.2.0.7713/public/mobilenet-ssd/mobilenet-ssd.caffemodel --data_type FP32 --scale 256 --mean_values [127,127,127] -o /app/resources/FP32
RUN mo --input_model /opt/intel/openvino_2022.2.0.7713/public/mobilenet-ssd/mobilenet-ssd.caffemodel --data_type FP16 --scale 256 --mean_values [127,127,127] -o /app/resources/FP16

RUN cd /app/resources && \
    wget -O face-demographics-walking.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4 && \
    wget -O bottle-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4 && \
    wget -O head-pose-face-detection-female.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4
COPY . /app
COPY startup.sh /startup.sh
RUN chmod a+x /startup.sh

CMD ["/startup.sh"]
