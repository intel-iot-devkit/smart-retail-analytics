#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import math
import time
from collections import namedtuple
from argparse import ArgumentParser
import logging as log
import numpy as np
from inference import Network
from influxdb import InfluxDBClient
from flask import Flask, render_template, Response
import cv2
import json

# Constants
CONFIG_FILE = "../resources/config.json"
MAX_FRAME_GONE = 3
INTEREST_COUNT_TIME = 5
SENTIMENT_LABEL = ['neutral', 'happy', 'sad', 'surprise', 'anger']
IPADDRESS = "localhost"
PORT = 8086
DATABASE_NAME = "Retail_Analytics"
CENTROID_DISTANCE = 150

# Global variables
check_feed_type = [False, False, False]  # [shopper, traffic, shelf]
centroids = []
tracked_person = []
person_id = 0
interested = 0
not_interested = 0
db_client = None
myriad_plugin = None
Point = namedtuple("Point", "x,y")
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True
template_dir = os.path.abspath('../templates')


class Centroid:
    """
    Store centroid details of the face detected for tracking
    """
    def __init__(self, p_id, point, gone_count):
        self.id = p_id
        self.point = point
        self.gone_count = gone_count


class Person:
    """
    Store the data of the people for tracking
    """
    def __init__(self, p_id, in_time):
        self.id = p_id
        self.counted = False
        self.gone = False
        self.in_time = in_time
        self.out_time = None
        self.looking = 0
        self.positive = 0
        self.negative = 0
        self.neutral = 0
        self.sentiment = ''


class VideoCap:
    """
    Store the data and manage multiple input video feeds
    """
    def __init__(self, input_name, input_number, feed_type, labels=None):
        self.vc = cv2.VideoCapture(input_name)
        self.input_number = input_number
        self.type = feed_type
        self.infer_network = None
        self.utime = time.time()
        self.nchw = []
        self.curr_req = 0
        self.next_req = 1
        if self.type == 'shopper':
            self.nchw_hp = []
            self.nchw_md = []
            self.thresh = 0.7

        if self.type == 'shelf' or self.type == 'traffic':
            self.thresh = 0.145
            self.labels = labels
            self.labels_map = []
            self.last_correct_count = [0] * len(self.labels)
            self.total_count = [0] * len(self.labels)
            self.current_count = [0] * len(self.labels)
            self.changed_count = [False] * len(self.labels)
            self.candidate_count = [0] * len(self.labels)
            self.candidate_confidence = [0] * len(self.labels)
            self.CONF_CANDIDATE_CONFIDENCE = 6
            if self.type == 'traffic':
                self.nchw_pr = []
                self.mog = cv2.createBackgroundSubtractorMOG2()
                self.CONF_CANDIDATE_CONFIDENCE = 3
                self.thresh = 0.45


def args_parser():
    """
    Parse command line arguments.

    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm", "--facemodel", help="Path to an .xml file with a pre-trained face detection model")
    parser.add_argument("-pm", "--posemodel", help="Path to an .xml file with a pre-trained model head pose model")
    parser.add_argument("-mm", "--moodmodel", help="Path to an .xml file with a pre-trained model "
                                                   "mood detection model")
    parser.add_argument("-om", "--objmodel", help="Path to an .xml file with a pre-trained model"
                                                  " object detection model")
    parser.add_argument("-pr", "--personmodel", help="Path to an .xml file with a pre-trained model "
                                                     "person detection model")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.")
    parser.add_argument("-d_fm", "--facedevice", default="CPU", type=str,
                        help="Specify the target device for face detection model to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. To run with multiple devices use "
                        "MULTI:<device1>,<device2>,etc. Looks" "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-d_pm", "--posedevice", default="CPU", type=str,
                        help="Specify the target device for head pose model to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. To run with multiple devices use "
                        "MULTI:<device1>,<device2>,etc. Looks" "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-d_mm", "--mooddevice", default="CPU", type=str,
                        help="Specify the target device for mood model to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. To run with multiple devices use "
                        "MULTI:<device1>,<device2>,etc. Looks" "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-d_om", "--objectdevice", default="CPU", type=str,
                        help="Specify the target device for object detection model to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. To run with multiple devices use "
                        "MULTI:<device1>,<device2>,etc. Looks" "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-d_pd", "--persondevice", default="CPU", type=str,
                        help="Specify the target device for person detection model to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. To run with multiple devices use "
                        "MULTI:<device1>,<device2>,etc. Looks" "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-lb", "--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-f", "--flag", help="sync or async", default="async", type=str)
    parser.add_argument("-ip", "--db_ipaddress", help="InfluxDB container IP address", default="localhost")
    args = parser.parse_args()
    return args


def check_args(args):
    """
    Validate the command line arguments

    :param args: Command line arguments
    :return: None
    """
    global check_feed_type
    global IPADDRESS

    if check_feed_type[0] is True:
        assert args.facemodel, 'Please specify the path to face detection model using the argument --facemodel or -fm'
        assert args.posemodel, 'Please specify the path to head pose model using the argument --posemodel or -pm'
        assert args.moodmodel, 'Please specify the path to mood detection model using the argument --moodmodel or -mm'

    if check_feed_type[1] is True:
        assert args.personmodel, 'Please specify the path to person detection model using the argument ' \
                                 '--personmodel or -pr'

    if check_feed_type[2] is True:
        assert args.objmodel, 'Please specify the path to object detection model using the argument --objmodel or -om'
        assert args.labels, 'Please specify the path label file using the argument --labels or -lb'

    IPADDRESS = args.db_ipaddress


def parse_conf_file():
    """
    Parse the configuration file and store the data in VideoCap object

    :return video_caps: List of VideoCap object containing input stream data
    """
    global CONFIG_FILE
    global check_feed_type

    print("in parse_conf_file()\n")
    video_caps = []

    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    for idx, item in enumerate(config['inputs']):
        labels = []
        parse_video = item['video']
        input_number = idx + 1
        if 'type' in item.keys():
            feed_type = item['type']
            if len(feed_type) == 0:
                print("Ignoring video {}... Format error".format(parse_video))
                continue
            if feed_type == 'shelf':
                check_feed_type[2] = True
                if 'label' in item.keys():
                    labels = [item['label']]
                    if len(labels) == 0:
                        print("Ignoring video {}... Format error".format(parse_video))
                        continue
                else:
                    print("Format error while reading labels for {}".format(feed_type))
                    continue
            elif feed_type == 'traffic':
                check_feed_type[1] = True
                labels = ['person']
            elif feed_type == 'shopper':
                check_feed_type[0] = True
            if parse_video.isdigit():
                video_cap = VideoCap(int(parse_video), input_number, feed_type, labels)
            else:
                assert os.path.isfile(parse_video), "{} doesn't exist".format(parse_video)
                video_cap = VideoCap(parse_video, input_number, feed_type, labels)
            video_cap.input_name = parse_video
            video_caps.append(video_cap)
        else:
            print("Feed type not specified for ", parse_video)

    for video_cap in video_caps:
        print("trying to evaluate video_cap.vc.isOpened()")
        print(str(video_cap))
        print(str(video_cap.vc))
        print(str(video_cap.vc.isOpened()))
        if not video_cap.vc.isOpened():
            print("{} is not initialized".format(video_cap.input_name))
        print(str(video_cap.input_name))
        assert video_cap.vc.isOpened(), "{} is not initialized".format(video_cap.input_name)
        video_cap.input_width = video_cap.vc.get(3)
        video_cap.input_height = video_cap.vc.get(4)
        if video_cap.type == 'traffic':
            video_cap.accumulated_frame = np.zeros(
                (int(video_cap.input_height), int(video_cap.input_width)), np.uint8)

    return video_caps


def load_model_device(infer_network, model, device, in_size, out_size, num_requests, cpu_extension, tag):
    """
    Loads the networks

    :param infer_network: Object of the Network() class
    :param model: .xml file of pre trained model
    :param device: Target device
    :param in_size: Number of input layers
    :param out_size: Number of output layers
    :param num_requests: Index of Infer request value. Limited to device capabilities
    :param cpu_extension: extension for the CPU device
    :return:  Shape of input layer
    """
    if 'MULTI' not in device and device not in accepted_devices:
        print("Unsupported device: " + device)
        sys.exit(1)
    elif 'MULTI' in device:
        target_devices = device.split(':')[1].split(',')
        for multi_device in target_devices:
            if multi_device not in accepted_devices:
                print("Unsupported device: " + device)
                sys.exit(1)

    global myriad_plugin
    if device == 'MYRIAD':
        if myriad_plugin is None:
            myriad_plugin, (nchw) = infer_network.load_model(model, device, in_size, out_size, num_requests)
        else:
            nchw = infer_network.load_model(model, device, in_size, out_size, num_requests, plugin=myriad_plugin)[1]
    else:
        nchw = infer_network.load_model(model, device, in_size, out_size, num_requests, cpu_extension, tag)[1]

    return nchw


def load_models(video_caps, args):
    """
    Load the required models

    :param video_caps: List of VideoCap objects
    :param args: Command line arguments
    :return: None
    """
    global check_feed_type

    # Check if one of the feed type is "shopper". If yes, load the face, head pose and mood detection model
    if check_feed_type[0]:
        infer_network_face = Network()
        infer_network_pose = Network()
        infer_network_mood = Network()
        tag_face = {"VPU_HDDL_GRAPH_TAG": "tagFace"}
        tag_pose = {"VPU_HDDL_GRAPH_TAG": "tagPose"}
        tag_mood = {"VPU_HDDL_GRAPH_TAG": "tagMood"}
        nchw_fd = load_model_device(infer_network_face, args.facemodel, args.facedevice, 1, 1, 2,
                                    args.cpu_extension, tag_face)
        nchw_hp = load_model_device(infer_network_pose, args.posemodel, args.posedevice, 1, 3, 2,
                                    args.cpu_extension, tag_pose)
        nchw_md = load_model_device(infer_network_mood, args.moodmodel, args.mooddevice, 1, 1, 2,
                                    args.cpu_extension, tag_mood)

    if check_feed_type[2]:
        infer_network = Network()
        tag_obj = {"VPU_HDDL_GRAPH_TAG": "tagMobile"}
        nchw = load_model_device(infer_network, args.objmodel, args.objectdevice, 1, 1, 2, args.cpu_extension, tag_obj)

    if check_feed_type[1]:
        infer_network_person = Network()
        tag_person = {"VPU_HDDL_GRAPH_TAG": "tagPerson"}
        nchw_pr = load_model_device(infer_network_person, args.personmodel, args.persondevice, 2, 1, 2,
                                    args.cpu_extension, tag_person)

    for video_cap in video_caps:
        if video_cap.type == 'shopper':
            video_cap.infer_network = infer_network_face
            video_cap.infer_network_hp = infer_network_pose
            video_cap.infer_network_md = infer_network_mood
            video_cap.nchw.extend(nchw_fd)
            video_cap.nchw_hp.extend(nchw_hp)
            video_cap.nchw_md.extend(nchw_md)

        if video_cap.type == 'shelf':
            video_cap.infer_network = infer_network
            video_cap.nchw.extend(nchw)

        if video_cap.type == 'traffic':
            video_cap.infer_network = infer_network_person
            video_cap.nchw.extend(nchw_pr)


def object_detection(video_cap, res):
    """
    Parse the inference result to get the detected object

    :param video_cap: VideoCap object of the frame on which the object is detected
    :param res: Inference output
    :return obj_det: List of coordinates of bounding boxes of the objects detected
    """
    obj_det = []

    for obj in res[0][0]:
        label = int(obj[1]) - 1

        # Draw only objects when probability is more than specified threshold
        if obj[2] > video_cap.thresh:
            # If the feed type is traffic shelf, look only for the person
            if  video_cap.type == 'traffic' and label == 0:
                video_cap.current_count[label] += 1

            # If the feed type is traffic shelf, look only for the objects that are specified by the user
            if video_cap.type == 'shelf':
                if label not in video_cap.labels_map:
                    continue
                label_idx = video_cap.labels_map.index(label)
                video_cap.current_count[label_idx] += 1

            if obj[3] < 0:
                obj[3] = 0
            if obj[4] < 0:
                obj[4] = 0

            xmin = int(obj[3] * video_cap.input_width)
            ymin = int(obj[4] * video_cap.input_height)
            xmax = int(obj[5] * video_cap.input_width)
            ymax = int(obj[6] * video_cap.input_height)
            obj_det.append([xmin, ymin, xmax, ymax])
    return obj_det


def get_used_labels(video_caps, args):
    """
    Read the model's label file and get the position of labels required by the application

    :param video_caps: List of VideoCap objects
    :param args: Command line arguments
    :return labels: List of labels present in the label file
    """
    global check_feed_type

    if check_feed_type[1] is False and check_feed_type[2] is False:
        return

    assert os.path.isfile(args.labels), "{} file doesn't exist".format(args.labels)
    with open(args.labels, 'r') as label_file:
        labels = [x.strip() for x in label_file]

    assert labels != [], "No labels found in {} file".format(args.labels)
    for video_cap in video_caps:
        if video_cap.type == 'shelf' or video_cap.type == 'traffic':
            for label in video_cap.labels:
                if label in labels:
                    label_idx = labels.index(label)
                    video_cap.labels_map.append(label_idx)
                else:
                    video_cap.labels_map.append(False)

    return labels


def process_output(video_cap):
    """
    Count the number of object detected

    :param video_cap: VideoCap object
    :return: None
    """
    for i in range(len(video_cap.labels)):
        if video_cap.candidate_count[i] == video_cap.current_count[i]:
            video_cap.candidate_confidence[i] += 1
        else:
            video_cap.candidate_confidence[i] = 0
            video_cap.candidate_count[i] = video_cap.current_count[i]

        if video_cap.candidate_confidence[i] == video_cap.CONF_CANDIDATE_CONFIDENCE:

            video_cap.candidate_confidence[i] = 0
            video_cap.changed_count[i] = True
        else:
            continue
        if video_cap.current_count[i] > video_cap.last_correct_count[i]:
            video_cap.total_count[i] += video_cap.current_count[i] - video_cap.last_correct_count[i]
        video_cap.last_correct_count[i] = video_cap.current_count[i]


def remove_centroid(p_id):
    """
    Remove the centroid from the "centroids" list when the person is out of the frame and
    set the person.gone variable as true

    :param p_id: ID of the person whose centroid data has to be deleted
    :return: None
    """
    global centroids
    global tracked_person

    for idx, centroid in enumerate(centroids):
        if centroid.id is p_id:
            del centroids[idx]
            break

    if tracked_person[p_id]:
        tracked_person[p_id].gone = True
        tracked_person[p_id].out_time = time.time()


def add_centroid(point):
    """
    Add the centroid of the object to the "centroids" list

    :param point: Centroid point to be added
    :return: None
    """
    global person_id
    global centroids
    global tracked_person

    centroid = Centroid(person_id, point, gone_count=0)
    person = Person(person_id, time.time())
    centroids.append(centroid)
    tracked_person.append(person)
    person_id += 1


def closest_centroid(point):
    """
    Find the closest centroid

    :param point: Coordinate of the point for which the closest centroid point has to be detected
    :return p_idx: Id of the closest centroid
            dist: Distance of point from the closest centroid
    """
    global centroids
    p_idx = 0
    dist = sys.float_info.max

    for idx, centroid in enumerate(centroids):
        _point = centroid.point
        dx = point.x - _point.x
        dy = point.y - _point.y
        _dist = math.sqrt(dx*dx + dy*dy)
        if _dist < dist:
            dist = _dist
            p_idx = centroid.id

    return [p_idx, dist]


def update_centroid(points, looking, sentiment, fps):
    """
    Update the centroid data in the centroids list and check whether the person is interested or not interested

    :param points: List of centroids of the faces detected
    :param looking: List of bool values indicating if the person is looking at the camera or not
    :param sentiment: List containing the mood of the people looking at the camera
    :param fps: FPS of the input stream
    :return: None
    """
    global MAX_FRAME_GONE
    global INTEREST_COUNT_TIME
    global interested
    global not_interested
    global centroids
    global tracked_person

    if len(points) is 0:
        for idx, centroid in enumerate(centroids):
            centroid.gone_count += 1
            if centroid.gone_count > MAX_FRAME_GONE:
                remove_centroid(centroid.id)

    if not centroids:
        for idx, point in enumerate(points):
            add_centroid(point)
    else:
        checked_points = len(points) * [None]
        checked_points_dist = len(points) * [None]
        for idx, point in enumerate(points):
            p_id, dist = closest_centroid(point)
            if dist > CENTROID_DISTANCE:
                continue

            if p_id in checked_points:
                p_idx = checked_points.index(p_id)
                if checked_points_dist[p_idx] > dist:
                    checked_points[p_idx] = None
                    checked_points_dist[p_idx] = None

            checked_points[idx] = p_id
            checked_points_dist[idx] = dist

        for centroid in centroids:
            if centroid.id in checked_points:
                p_idx = checked_points.index(centroid.id)
                centroid.point = points[p_idx]
                centroid.gone_count = 0
            else:
                centroid.gone_count += 1
                if centroid.gone_count > MAX_FRAME_GONE:
                    remove_centroid(centroid.id)

        for idx in range(len(checked_points)):
            if checked_points[idx] is None:
                add_centroid(points[idx])
            else:
                if looking[idx] is True:
                    tracked_person[checked_points[idx]].sentiment = sentiment[idx]
                    tracked_person[checked_points[idx]].looking += 1
                    if sentiment[idx] == "happy" or sentiment[idx] == "surprise":
                        tracked_person[checked_points[idx]].positive += 1
                    elif sentiment[idx] == 'sad' or sentiment[idx] == 'anger':
                        tracked_person[checked_points[idx]].negative += 1
                    elif sentiment[idx] == 'neutral':
                        tracked_person[checked_points[idx]].neutral += 1
                else:
                    tracked_person[checked_points[idx]].sentiment = "Not looking"

        for person in tracked_person:
            if person.counted is False:
                positive = person.positive + person.neutral

                # If the person is looking at the camera for specified time
                # and his mood is positive, increment the interested variable
                if (person.looking > fps * INTEREST_COUNT_TIME) and (positive > person.negative):
                    interested += 1
                    person.counted = True

                    # If the person is gone out of the frame, increment the not_interested variable
                    if person.gone is True:
                        not_interested += 1
                        person.counted = True


def detect_head_pose_and_emotions(video_cap, object_det):
    """
    Detect the head pose and emotions of the faces detected

    :param video_cap: VideoCap object
    :param object_det: List of faces detected in the frame
    :return: None
    """

    global SENTIMENT_LABEL
    global centroids
    global is_async_mode

    frame_centroids = []
    looking = []
    sentiment = []

    for face in object_det:
        xmin, ymin, xmax, ymax = face

        # Find the centroid of the face
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + int(width / 2)
        y = ymin + int(height / 2)
        point = Point(x, y)
        frame_centroids.append(point)

        # Check the head pose
        if is_async_mode:
            head_pose = video_cap.next_frame[ymin:ymax, xmin:xmax]
            in_frame = cv2.resize(head_pose, (video_cap.nchw_hp[3], video_cap.nchw_hp[2]))
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((video_cap.nchw_hp[0], video_cap.nchw_hp[1],
                                         video_cap.nchw_hp[2], video_cap.nchw_hp[3]))

            video_cap.infer_network_hp.exec_net(video_cap.next_req, in_frame)
        else:
            head_pose = video_cap.frame[ymin:ymax, xmin:xmax]
            in_frame = cv2.resize(head_pose, (video_cap.nchw_hp[3], video_cap.nchw_hp[2]))
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((video_cap.nchw_hp[0], video_cap.nchw_hp[1],
                                         video_cap.nchw_hp[2], video_cap.nchw_hp[3]))

            video_cap.infer_network_hp.exec_net(video_cap.curr_req, in_frame)
        if video_cap.infer_network_hp.wait(video_cap.curr_req) == 0:

            # Parse head pose detection results
            angle_p_fc = video_cap.infer_network_hp.get_output(video_cap.curr_req, "angle_p_fc")
            angle_y_fc = video_cap.infer_network_hp.get_output(video_cap.curr_req, "angle_y_fc")

            # Check if the person is looking at the camera
            if (angle_y_fc > -22.5) & (angle_y_fc < 22.5) & (angle_p_fc > -22.5) & (angle_p_fc < 22.5):
                looking.append(True)

                # Find the emotions of the person
                in_frame = cv2.resize(head_pose, (video_cap.nchw_md[3], video_cap.nchw_md[2]))
                in_frame = in_frame.transpose((2, 0, 1))
                in_frame = in_frame.reshape((video_cap.nchw_md[0], video_cap.nchw_md[1],
                                             video_cap.nchw_md[2], video_cap.nchw_md[3]))
                if is_async_mode:
                    video_cap.infer_network_md.exec_net(video_cap.next_req, in_frame)
                else:
                    video_cap.infer_network_md.exec_net(video_cap.curr_req, in_frame)
                video_cap.infer_network_md.wait(video_cap.curr_req)
                res = video_cap.infer_network_md.get_output(video_cap.curr_req)
                emotions = np.argmax(res)
                sentiment.append(SENTIMENT_LABEL[emotions])
            else:
                looking.append(False)
                sentiment.append(-1)
        if is_async_mode:
            video_cap.curr_req, video_cap.next_req = video_cap.next_req, video_cap.curr_req
            video_cap.frame = video_cap.next_frame
    update_centroid(frame_centroids, looking, sentiment, video_cap.vc.get(cv2.CAP_PROP_FPS))
    for idx, centroid in enumerate(centroids):
        cv2.rectangle(video_cap.frame, (centroid.point.x, centroid.point.y),
                      (centroid.point.x + 1, centroid.point.y + 1), (0, 255, 0), 4, 16)
        cv2.putText(video_cap.frame, "person:{}".format(centroid.id), (centroid.point.x + 1, centroid.point.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def heatmap_generation(video_cap):
    """
    Generates the heatmap

    :param video_cap: VideoCap of input feed for which the heatmap has to be generated
    :return: None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(video_cap.frame, cv2.COLOR_BGR2GRAY)

    # Remove the background
    fgbgmask = video_cap.mog.apply(gray)

    # Threshold the image
    thresh = 2
    max_value = 2
    threshold_frame = cv2.threshold(fgbgmask, thresh, max_value, cv2.THRESH_BINARY)[1]

    # Add threshold image to the accumulated image
    video_cap.accumulated_frame = cv2.add(threshold_frame, video_cap.accumulated_frame)
    colormap_frame = cv2.applyColorMap(video_cap.accumulated_frame, cv2.COLORMAP_HOT)
    video_cap.frame = cv2.addWeighted(video_cap.frame, 0.6, colormap_frame, 0.4, 0)


def update_info_shopper(video_cap):
    """
    Send "shopper" data to InfluxDB

    :param video_cap: List of VideoCap object
    :return: None
    """
    global tracked_person
    global interested
    global not_interested
    global db_client
    json_body = [{"measurement": "{}_interest".format(video_cap.type),
                  "fields": {
                      "time": time.time(),
                      "Interested": interested,
                      "Not Interested": not_interested,
                      "Total Count": len(tracked_person)
                  }
                 }]
    db_client.write_points(json_body)
    for person in tracked_person:
        if person.gone is False:
            tm = time.time() - person.in_time
            looking_time = person.looking/video_cap.vc.get(cv2.CAP_PROP_FPS)
            json_body = [{
                "measurement": "{}_duration".format(video_cap.type),
                "fields": {
                    "person": person.id,
                    "Looking time": looking_time,
                    "Time in frame": tm,
                    "Current Mood": person.sentiment
                }
            }]
        db_client.write_points(json_body)


def update_info_object(labels, video_cap):
    """
    Send "traffic" and "shelf" data to InfluxDB

    :param labels: List of labels present in label file
    :param video_cap: VideoCap object
    :return: None
    """
    global db_client
    for idx, label in enumerate(video_cap.labels_map):
        json_body = [
            {
                "measurement": video_cap.type,
                "tags": {
                    "object": labels[label],
                },
                "fields": {
                    "time": time.time(),
                    "Current Count": video_cap.current_count[idx],
                    "Total Count": video_cap.total_count[idx],
                }
            }]
        db_client.write_points(json_body)


def create_database():
    """
    Connect to InfluxDB and create the database

    :return: None
    """
    global db_client

    print("create_database()\n")

    proxy = {"http": "http://{}:{}".format(IPADDRESS, PORT)}
    db_client = InfluxDBClient(host=IPADDRESS, port=PORT, proxies=proxy, database=DATABASE_NAME)
    db_client.create_database(DATABASE_NAME)


def retail_analytics():
    """
    Detect objects on multiple input video feeds and process the output

    :return: None
    """
    global centroids
    global tracked_person
    global db_client

    print("in retail_analytics()\n")
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    video_caps = parse_conf_file()
    print("before asserting input source\n")
    assert len(video_caps) != 0, "No input source given in Configuration file"
    print("after asserting input source\n")
    args = args_parser()
    check_args(args)
    load_models(video_caps, args)
    labels = get_used_labels(video_caps, args)
    print(" before create_database\n")
    create_database()
    print("after create_database\n")
    min_fps = min([i.vc.get(cv2.CAP_PROP_FPS) for i in video_caps])
    no_more_data = [False] * len(video_caps)
    frames = [None] * len(video_caps)
    start_time = time.time()
    global is_async_mode
    if args.flag == "async":
        is_async_mode = True
        print('Application running in async mode')
    else:
        is_async_mode = False
        print('Application running in sync mode')
    cur_request_id_tr = 0
    next_request_id_tr = 1
    cur_request_id_sf = 0
    next_request_id_sf = 1
    cur_request_id_sh = 0
    next_request_id_sh = 1
    det_time = 0
    input_blob = ["data", "im_info"]
    cur_request_id = 0
    print("before main loop in retail_analytics()\n")
    # Main loop for object detection in multiple video streams
    while True:
        print("iteration of main loop in retail_analytics()\n")
        for idx, video_cap in enumerate(video_caps):
            vfps = int(round(video_cap.vc.get(cv2.CAP_PROP_FPS)))
            for i in range(0, int(round(vfps / min_fps))):
                if is_async_mode:
                    ret, video_cap.next_frame = video_cap.vc.read()
                else:
                    ret, video_cap.frame = video_cap.vc.read()

                # If no new frame or error in reading the frame, exit the loop
                if not ret:
                    no_more_data[idx] = True
                    break

                if video_cap.type == 'traffic' or video_cap.type == 'shelf':
                    video_cap.current_count = [0] * len(video_cap.labels)
                    video_cap.changed_count = [False] * len(video_cap.labels)
                inf_start = time.time()
                if is_async_mode:
                    in_frame = cv2.resize(video_cap.next_frame, (video_cap.nchw[3], video_cap.nchw[2]))
                    in_frame = in_frame.transpose((2, 0, 1))
                    in_frame = in_frame.reshape((video_cap.nchw[0], video_cap.nchw[1],
                                                 video_cap.nchw[2], video_cap.nchw[3]))
                    if video_cap.type == 'traffic':
                        video_cap.infer_network.exec_net(next_request_id_tr, in_frame,
                                                         input_blob, video_cap.vc.get(3), video_cap.vc.get(4))
                        cur_request_id = cur_request_id_tr
                    elif video_cap.type == 'shelf':
                        video_cap.infer_network.exec_net(next_request_id_sf, in_frame)
                        cur_request_id = cur_request_id_sf
                    else:
                        video_cap.infer_network.exec_net(next_request_id_sh, in_frame)
                        cur_request_id = cur_request_id_sh
                    video_cap.frame = video_cap.next_frame
                else:
                    in_frame = cv2.resize(video_cap.frame, (video_cap.nchw[3], video_cap.nchw[2]))
                    in_frame = in_frame.transpose((2, 0, 1))
                    in_frame = in_frame.reshape((video_cap.nchw[0], video_cap.nchw[1],
                                                 video_cap.nchw[2], video_cap.nchw[3]))
                    if video_cap.type == 'traffic':
                        video_cap.infer_network.exec_net(cur_request_id_tr, in_frame, input_blob,
                                                         video_cap.vc.get(3), video_cap.vc.get(4))
                    elif video_cap.type == 'shelf':
                        video_cap.infer_network.exec_net(cur_request_id_sf, in_frame)
                    else:
                        video_cap.infer_network.exec_net(cur_request_id_sh, in_frame)
                if video_cap.infer_network.wait(cur_request_id) == 0:
                    inf_end = time.time()
                    det_time = inf_end - inf_start

                    # Pass the frame to the inference engine and get the results
                    res = video_cap.infer_network.get_output(cur_request_id)

                    # Process the result obtained from the inference engine
                    object_det = object_detection(video_cap, res)

                    # If the feed type is "traffic" or "shelf", check the current and total count of the object
                    if video_cap.type == 'traffic' or video_cap.type == 'shelf':
                        process_output(video_cap)

                        # If feed type is "traffic", generate the heatmap
                        if video_cap.type == 'traffic':
                            heatmap_generation(video_cap)

                        # Send the data to InfluxDB
                        if time.time() >= video_cap.utime + 1:
                            update_info_object(labels, video_cap)
                            video_cap.utime = time.time()

                    else:
                        # Detect head pose and emotions of the faces detected
                        detect_head_pose_and_emotions(video_cap, object_det)

                        # Send the data to InfluxDB
                        if time.time() >= video_cap.utime + 1:
                            update_info_shopper(video_cap)
                            video_cap.utime = time.time()
                if is_async_mode:
                    if video_cap.type == 'traffic':
                        cur_request_id_tr, next_request_id_tr = next_request_id_tr, cur_request_id_tr
                    elif video_cap.type == 'shopper':
                        cur_request_id_sh, next_request_id_sh = next_request_id_sh, cur_request_id_sh
                    else:
                        cur_request_id_sf, next_request_id_sf = next_request_id_sf, cur_request_id_sf

                fps_time = time.time() - start_time
                fps_message = "FPS: {:.3f} fps".format(1 / fps_time)
                start_time = time.time()
                inf_time_message = "Inference time: N\A for async mode" if is_async_mode else\
                    "Inference time: {:.3f} ms".format(det_time * 1000)
                cv2.putText(video_cap.frame, inf_time_message, (10, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (200, 10, 10), 1)
                cv2.putText(video_cap.frame, fps_message, (10, int(video_cap.input_height) - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            # If no new frame, continue to the next input feed
            if no_more_data[idx] is True:
                continue

            # Print the results on the frame and stream it
            message = "Feed Type: {}".format(video_cap.type)
            cv2.putText(video_cap.frame, message, (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            if video_cap.type == 'traffic' or video_cap.type == 'shelf':
                ht = 50
                for indx, label in enumerate(video_cap.labels_map):
                    message = "{} -> Total Count: {}, Current Count: {}".format(labels[label],
                                                                                video_cap.total_count[indx],
                                                                                video_cap.current_count[indx])
                    cv2.putText(video_cap.frame, message, (10, ht), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                    ht += 20
            else:
                message = "Face -> Total Count: {}, Current Count: {}".format(len(tracked_person), len(centroids))
                cv2.putText(video_cap.frame, message, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                ht = 75
                for person in tracked_person:
                    if person.gone is False:
                        message = "Person {} is {}".format(person.id, person.sentiment)
                        cv2.putText(video_cap.frame, message, (10, ht), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (200, 10, 10), 1)
                        ht += 20

            frames[idx] = video_cap.frame

        # Resize the processed frames to stream on Grafana
        for idx, img in enumerate(frames):
            frames[idx] = cv2.resize(img, (480, 360))

        # Encode the frames into a memory buffer.
        ret, img = cv2.imencode('.jpg', np.hstack(frames))
        img = img.tobytes()

        # Yield the output frame to the server
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')

        # If no more frames, exit the loop
        if False not in no_more_data:
            break


# Create object for Flask class
app = Flask(__name__, template_folder=template_dir)
app.logger.disabled = False
log_ = log.getLogger('werkzeug')
log_.disabled = False

@app.route('/')
def index():
    """
    Trigger the index() function on opening "0.0.0.0:5000/" URL
    :return: html file
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Trigger the video_feed() function on opening "0.0.0.0:5000/video_feed" URL
    :return:
    """
    try:
        return Response(retail_analytics(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("caught exception " + str(e) + " in video_feed()\n")
        raise


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0',debug=True,use_reloader=False)
    except Exception as e:
        print("caught exception " + str(e) + " in app.run\n")
        raise
