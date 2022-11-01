#!/bin/bash
set -euxo pipefail

cd /opt/intel/openvino_2022.2.0.7713/intel/face-detection-adas-0001/FP32/
SERVER=sra-nginx-service.default.svc.cluster.local
wget http://$SERVER/face-detection-adas-0001-FP32/face-detection-adas-0001.bin
wget http://$SERVER/face-detection-adas-0001-FP32/face-detection-adas-0001.xml

cd /app/application
python3 smart_retail_analytics.py -fm /opt/intel/openvino_2022.2.0.7713/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -pm /opt/intel/openvino_2022.2.0.7713/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -mm /opt/intel/openvino_2022.2.0.7713/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -om ../resources/FP32/mobilenet-ssd.xml -pr /opt/intel/openvino_2022.2.0.7713/intel/person-detection-retail-0002/FP32/person-detection-retail-0002.xml -lb /app/resources/labels.txt -ip localhost

