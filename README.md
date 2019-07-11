# Smart Retail Analytics

| Details           |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  Python* 3.5 |
| Time to Complete:    |  50-70min     |

![Smart Retail Analytics](./docs/images/retail-analytics.png)

An application capable of detecting objects on any number of screens.

## What it Does
This application is one of a series of IoT reference implementations aimed at instructing users on how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. This reference implementation monitors people activity inside a retail store and keeps a check on the inventory.


## Requirements
### Hardware
* 6th to 8th Generation Intel® Core™ processors with Iris® Pro graphics or Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)<br>
   *Note*: We recommend using a 4.14+ Linux* kernel with this software. Run the following command to determine your kernel version:
   ```
   uname -a
   ```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit 2019 R1 release 
* Grafana* v5.3.2 
* InfluxDB* v1.6.2


## How it Works
The application uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. It accepts multiple video input feeds and user can specify the feed type for each video. 
There are three feed types that application supports:
* Shopper: If the feed type of the video is shopper, the application grabs the frame from that input stream and uses a Deep Neural Network model for detecting the faces in it. If there is anybody present in the frame, it is counted as a shopper. Once the face is detected, the application uses head-pose estimation model to check the head pose of the person. If the person is looking at the camera then his emotions are detected using emotions recognition model. Using the data obtained from this, it infers if the person is interested or not and gives the total number of people detected. It also measures the duration for which the person is present in the frame and the duration for which he was looking at the camera.

* Store traffic: If the video feed type is traffic, the application uses a Deep Neural Network model to detect people in the frame. The total number of people visited and the number of people currently present in front the camera is obtained from this.

* Shelf: This feed type is used to keep a check on the product inventory. If the video feed type is shelf, an object detection model is used to detect the product specified by the user in the frame from this video stream. It detects the objects and gives the number of objects present in the frame.

The application is capable of processing multiple video input feeds, each having different feed type. The data obtained from these videos is store in InfluxDB for analysis and visualized on Grafana. It used Flask Python web framework to live stream the output videos to the Grafana.


![Retail Analytics](./docs/images/architectural-diagram.png)



## Setup

### Install the Intel® Distribution of OpenVINO™ toolkit
Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux on how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime Package if you plan to run inference on the GPU. It is not mandatory for CPU inference.

## Install the dependencies
#### InfluxDB* 

Use the commands below to install InfluxDB:
```
sudo apt install curl
sudo curl -sL https://repos.influxdata.com/influxdb.key | sudo apt-key add - 
source /etc/lsb-release
echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
sudo apt-get update 
sudo apt-get install influxdb
sudo service influxdb start
```

#### Grafana*

Use the commands below to install Grafana:

```
wget https://s3-us-west-2.amazonaws.com/grafana-releases/release/grafana_5.3.2_amd64.deb
sudo apt-get install -y adduser libfontconfig
sudo dpkg -i grafana_5.3.2_amd64.deb
sudo /bin/systemctl start grafana-server
```

Install the AJAX panel for grafana:
```
sudo grafana-cli plugins install ryantxu-ajax-panel
```

#### Install Python* Package Dependencies
```
sudo apt-get install python3-pip
pip3 install influxdb numpy flask
```

## Configure the application

The application uses three Intel® Pre-Trained models in the feed type `shopper` i.e. face detection model, head pose estimation model and emotion recognition model that can be downloaded using **model downloader** script. 

To download these models:
* Go to the **model_downloader** directory using the following command:
   ```
   cd /opt/intel/openvino/deployment_tools/tools/model_downloader
   ```
* Specify which model to download using the argument __--name__ :
   ```
   sudo ./downloader.py --name face-detection-adas-0001
   sudo ./downloader.py --name head-pose-estimation-adas-0001
   sudo ./downloader.py --name emotions-recognition-retail-0003
   ```

* To download the model for FP16, run the following commands:
   ```
   sudo ./downloader.py --name face-detection-adas-0001-fp16
   sudo ./downloader.py --name head-pose-estimation-adas-0001-fp16
   sudo ./downloader.py --name emotions-recognition-retail-0003-fp16
   ```

* These models will be downloaded in the locations given below: 
   * **face-detection**: /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/
   * **head-pose-estimation**: /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/

   * **emotions-recognition**: /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/

<br>
For video feed types __traffic__ and __shelf__, mobilenet-ssd model is used that can be downloaded using `downloader` script present in Intel® Distribution of OpenVINO™ toolkit. Instructions to download the mobilenet-ssd model is given below.


#### Download the mobilenet-ssd Model
* Go to the `model_downloader` directory present inside Intel® Distribution of OpenVINO™ toolkit install directory:
  ```
  cd /opt/intel/openvino/deployment_tools/tools/model_downloader/
  ```

* Specify which model to download with `--name` and the output path with `-o`; otherwise, the model will be downloaded to the current folder. Run the model downloader with the following command:
  ```
  sudo ./downloader.py --name mobilenet-ssd
  ```
  
* The model will be downloaded inside the `object_detection/common` directory. To make it work with the Intel® Distribution of OpenVINO™ toolkit, the model needs to be passed through the model optimizer to generate the IR (the .xml and .bin files). 

  **Note:** If you haven't configured the **model optimizer** yet, follow the instructions to configure it provided [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Config_Model_Optimizer.html).   

* After configuring the model optimizer, go to the **model optimizer** directory:
  ```
  cd /opt/intel/openvino/deployment_tools/model_optimizer/
  ```
   
* Run this command to optimize mobilenet-ssd:
  ```
  ./mo_caffe.py --input_model /opt/intel/openvino/deployment_tools/tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  -o $HOME/retail-analytics/resources/FP32 --data_type FP32 --scale 256 --mean_values [127,127,127]
  ```
  **Note:** Replace $HOME in the above command with the path to the _retail-analytics_ directory.

* To optimize the model for FP16:
  ```
  ./mo_caffe.py --input_model /opt/intel/openvino/deployment_tools/tools/model_downloader/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel  -o $HOME/retail-analytics/resources/FP16 --data_type FP16 --scale 256 --mean_values [127,127,127]
  ```

### The config file
The **resources/conf.txt** contains the videos along with the video feed type.  
Each block in the file contains video file name and type.<br> 
For example:
```
video: path-to-video
type: video-feed-type
```
The `path-to-video` is the path, on the local system, to a video to use as input.

If the video type is shelf, then the labels of the class (person, bottle, etc.) to be detected on that video is provided in the next line. The labels used in the _conf.txt_ file must be present in the labels from the _labels_ file.<br> 
For example:
```
video: ./resources/head-pose-face-detection-female.mp4
type: shopper

video: ./resources/face-demographics-walking.mp4
type: traffic

video: ./resources/bottle-detection.mp4
type: shelf
label: bottle
```

The application can use any number of videos for detection (i.e. the _conf.txt_ file can have any number of blocks), but the more videos the application uses in parallel, the more the frame rate of each video scales down. This can be solved by adding more computation power to the machine the application is running on.

### The labels file
The shelf feed type in the application requires a _labels_ file associated with the model being used for detection. All detection models work with integer labels and not string labels (e.g. for the ssd300 and mobilenet-ssd models, the number 15 represents the class "person"), that is why each model must have a _labels_ file, which associates an integer (the label the algorithm detects) with a string (denoting the human-readable label).   
The _labels_ file is a text file containing all the classes/labels that the model can recognize, in the order that it was trained to recognize them (one class per line).<br> 
For mobilenet-ssd model, _labels.txt_ file is provided in the _resources_ directory.


### What input video to use
The application works with any input video. Sample videos for object detection are provided [here](https://github.com/intel-iot-devkit/sample-videos/).  <br>

For first-use, we recommend using the [face-demographics-walking](https://github.com/intel-iot-devkit/sample-videos/blob/master/face-demographics-walking.mp4), [head-pose-face-detection-female](https://github.com/intel-iot-devkit/sample-videos/blob/master/head-pose-face-detection-female.mp4), [bottle-detection](https://github.com/intel-iot-devkit/sample-videos/blob/master/bottle-detection.mp4) videos. For example:

Go to _retail-analytics_ directory and run the following commands to download the videos:
```
cd resources
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4 
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4 
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4 	
cd .. 
```
The videos are downloaded in the `resources/` folder.


### Using camera stream instead of the video file
Replace `path/to/video` with the camera ID in conf.txt and the label to be found, where the ID is taken from the video device (the number X in /dev/videoX).
On Ubuntu, to list all available video devices use the following command:
```
ls /dev/video*
```
For example, if the output of above command is `/dev/video0`, then conf.txt would be:

```
video: 0
type: shopper
```

## Setup the environment
You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

## Run the application
To run the application with the required models:
```
python3 main.py -fm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml -mm /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml -om ./resources/FP32/mobilenet-ssd.xml -lb ./resources/labels.txt -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so
```
Once the command is executed in the terminal, configure the Grafana dashboard using the instructions given in the next section to see the output.

 ## Running on different hardware 

The application can use different hardware accelerator for different models. The user can specify the target device for each model using the command line argument as below:
* `-d_fm <device>`: Target device for Face Detection network (CPU, GPU, MYRIAD or HETERO:HDDL,CPU). 
* `-d_pm <device>`: Target device for Head Pose Estimation network (CPU, GPU, MYRIAD or HETERO:HDDL,CPU).
* `-d_mm <device>`: Target device for Emotions Recognition network (CPU, GPU, MYRIAD or HETERO:HDDL,CPU).
* `-d_om <device>`: Target device for mobilenet-ssd network (CPU, GPU, MYRIAD or HETERO:HDDL,CPU).



__For example:__<br>
To run Face Detection model with FP16 and Emotions Recognition model with FP32 on GPU, Head Pose Estimation model on MYRIAD and mobilenet-ssd on CPU, use the below command:
```
python3 main.py -fm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -pm /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -mm /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml -om ./resources/FP32/mobilenet-ssd.xml -lb ./resources/labels.txt -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d_fm GPU -d_pm MYRIAD -d_mm GPU -d_om CPU
```

By default, the application runs on CPU.<br>
**Note:** The Intel® Neural Compute Stick 2 and HDDL-R can only run FP16 models. The model that is passed to the application, must be of data type FP16. 


### Visualization on Grafana*

1. Start the Grafana server:

   ```
   sudo service grafana-server start
   ```

2. In your browser, go to [localhost:3000](http://localhost:3000).

3. Log in with user as **admin** and password as **admin**.

4. Click on **Configuration**.

5. Select **“Data Sources”**.

6. Click on **“+ Add data source”** and provide inputs below.

   - *Name*: Retail_Analytics
   - *Type*: InfluxDB
   - *URL*: http://localhost:8086
   - *Database*: Retail_Analytics
   - Click on “Save and Test”

   ![Retail Analytics](./docs/images/grafana1.png)

7. Click on **+** icon present on the left side of the browser, select **import**.

8. Click on **Upload.json File**.

9. Select the file name __retail-analytics.json__ from retail-analytics directory.

10. Select "Retail_Analytics" in **Select a influxDB data source**. 

    ![Retail Analytics](./docs/images/grafana2.png)

11. Click on import.


## Containerize the Application

To containerize the retail-analytics application using docker container, follow the instruction provided [here](./docker).
