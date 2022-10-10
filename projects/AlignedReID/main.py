import os
import torch
import numpy as np
import cv2
from detection import Detection
import argparse
import torchvision.ops.boxes as bops
from Alignedreid_demo import Aligned_Reid_class
from threading import Thread
from ReIdentification_module import Reid_module

# Arguments set up..
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--file_path', type=str,
                help='path to input video file')
args = vars(ap.parse_args())
if not args.get('file_path', False):
    print("please provide file path...")
else:
    path_received = args['file_path']
    with open(path_received,'r') as f:
        streams = f.readlines()

# Declarations..
stream_Array = [cv2.VideoCapture(src.strip()) for src in streams]
frame_width = int(stream_Array[0].get(3)/2)
frame_height = int(stream_Array[0].get(4)/2)
new_size = (frame_width,frame_height)
yolo = Detection()
tracker = cv2.TrackerCSRT_create()
init_bb = None
temp = '../AlignedReID/suspect_data/'
selected_src = None
result_dict = {}
csrt_box = None
key = None
is_suspect_identified = False
is_suspect_data_available = False
has_suspect_exit = False
result_dict = {}
path = '../AlignedReID/'
num_streams_srcs = len(stream_Array)
streams = [stream_name.strip() for stream_name in streams]
rd_module = Reid_module(streams)
frame_count = 0

def is_available(iou_vector, count):
    flag = True
    for item in iou_vector:
        if item != 0:
            flag = False
            break
    if flag is True:
        count += 1
        return flag, count
    return flag, count

def iou_check(yolo_bbox, csrt_box):
    csrt_x, csrt_y, csrt_w, csrt_h = csrt_box
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
    box1 = torch.tensor([[yolo_x1, yolo_y1, yolo_x2, yolo_y2]], dtype=torch.float)
    box2 = torch.tensor([[csrt_x, csrt_y, csrt_x + csrt_w, csrt_y + csrt_h]], dtype=torch.float)
    IOU = bops.box_iou(box1, box2)
    return IOU


def iou_check2(yolo_bbox, csrt_box):
    csrt_x1, csrt_y1, csrt_x2, csrt_y2 = csrt_box
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
    box1 = torch.tensor([[yolo_x1, yolo_y1, yolo_x2, yolo_y2]], dtype=torch.float)
    box2 = torch.tensor([[csrt_x1, csrt_y1, csrt_x2, csrt_y2]], dtype=torch.float)
    IOU = bops.box_iou(box1, box2)
    return IOU

def preprocessing(box, current_frame):
    x_shape, y_shape = current_frame.shape[1], current_frame.shape[0]
    x1, y1, x2, y2 = int(box[0] * x_shape), int(box[1] * y_shape), int(box[2] * x_shape), int(box[3] * y_shape)
    return [x1, y1, x2, y2]


def annotate_frame(frame, bboxes):
    for idx in range(len(bboxes)):
        cord = preprocessing(bboxes[idx], frame)
        frame = cv2.rectangle(frame, (cord[0], cord[1]), (cord[2], cord[3]), (0, 252, 124), 2)
    return frame


def crop_image(box, frame, index, dir_path,b_boxes, temp):
    feature_flag = False
    status = False
    features_reid_score = []
    Aligned_Ried = Aligned_Reid_class()
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    x1, y1, x2, y2 = int(box[0] * x_shape), int(box[1] * y_shape), int(box[2] * x_shape), int(box[3] * y_shape)
    b_box = [x1, y1, x2, y2]
    image = frame[y1:y2, x1:x2]
    image_name = 'image_' + str(index) + '.jpg'
    img_path = dir_path  + image_name
    cv2.imwrite(img_path, image)
    distance = Aligned_Ried.compute_distance(os.path.join(temp, 'suspect.jpg'), img_path)
    result_list = [status, b_box, distance]

    feature_flag = os.path.isfile(os.path.join(temp, 'feature_1.jpg'))

    if feature_flag is True:
        for i, b in enumerate(b_boxes):
            b = preprocessing(b, frame)
            IOU = iou_check2(b, b_box)

            if IOU * 100 > 0 and IOU != 1 and result_list[index][2] * 100 <= 70:
                x1, y1, x2, y2 = b
                image = frame[y1:y2, x1:x2]
                image_name = str(IOU * 100) + str(i) + '.jpg'
                img_path = dir_path  + image_name
                cv2.imwrite(img_path, image)
                distance = Aligned_Ried.compute_distance(os.path.join(temp, 'feature_1.jpg'), img_path)
                if distance * 100 <= 70:
                    features_reid_score.append(distance)

    if len(features_reid_score) != 0:
        arr = np.array(features_reid_score)
        result_list.append(np.min(arr))

    if len(result_list) == 4:
        join_reid_score = result_list[2] * result_list[3]
    else:
        join_reid_score = result_list[2]

    result_dict[index] += [result_list]
    if join_reid_score * 100 <= 45:
        idx = len(result_dict[index])-1
        result_dict[index][idx][0] = True

def call_to_reid_module(thread_id, frame, predictions, path, temp):
    newpath = path +  'src_data' + str(thread_id) + '/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for idx in range(len(predictions)):
        crop_image(predictions[idx], frame,thread_id,newpath,predictions, temp)

while True:
    is_suspect_data_available = os.path.isfile(os.path.join(temp, 'suspect.jpg'))
    frames_from_all_streams = [cv2.resize(stream_Array[idx].read()[1], new_size) for idx in range(len(stream_Array))]
    predictions_on_frames = [yolo.detect(frames_from_all_streams[idx])[1] for idx in range(len(frames_from_all_streams))]

    for idx in range(len(stream_Array)):
        result_dict[idx] = []
        frames_from_all_streams[idx] = annotate_frame(frames_from_all_streams[idx], predictions_on_frames[idx])
    if frame_count%10 ==0:
        if is_suspect_data_available is True and has_suspect_exit is True:
            Threads = [None] * num_streams_srcs
            for idx in range(num_streams_srcs):
                Threads[idx] = Thread(target=(call_to_reid_module), args=(idx, frames_from_all_streams[idx], predictions_on_frames[idx], path, temp) )
                Threads[idx].start()
                Threads[idx].join()
            for idx in range(num_streams_srcs):
                for jdx in range(len(result_dict[idx])):
                    if result_dict[idx][jdx][0] is True:
                        init_bb = result_dict[idx][jdx][1]
                        init_bb[2] = init_bb[2] - init_bb[0]
                        init_bb[3] = init_bb[3] - init_bb[1]
                        init_bb = tuple(init_bb)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frames_from_all_streams[idx], init_bb)
                        has_suspect_exit = False
                        selected_src = idx

            for idx in range(num_streams_srcs):
                Threads.pop(-1)

    if init_bb is not None:
        (success, csrt_box) = tracker.update(frames_from_all_streams[selected_src])
        iou_vector = [None] * len(predictions_on_frames[selected_src])

        cord = csrt_box
        if success:
            idx = 0
            for bbox in predictions_on_frames[selected_src]:
                cord = preprocessing(bbox, frames_from_all_streams[selected_src])
                IOU = iou_check(cord,csrt_box)*100
                iou_vector[idx] = int(IOU)
                if IOU >= 50:
                    suspect_cord = cord
                    suspect_cord[2] = suspect_cord[2] - suspect_cord[0]
                    suspect_cord[3] = suspect_cord[3] - suspect_cord[1]
                    suspect_cord = tuple(suspect_cord)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frames_from_all_streams[selected_src], suspect_cord)
                    (success, csrt_box) = tracker.update(frames_from_all_streams[selected_src])
                    no_suspect_count = 0
                idx += 1

            x, y, w, h = [int(x) for x in csrt_box]
            frames_from_all_streams[selected_src] = cv2.rectangle(frames_from_all_streams[selected_src], (x, y), (x + w, y + h), (255, 20, 255), 2)

    # we need to check whether the suspect is still in the currently selected stream...
    if selected_src is not None:
        is_roi_not_present, no_suspect_count = is_available(iou_vector, no_suspect_count)
        if is_roi_not_present is True and no_suspect_count > 2:
            has_suspect_exit = True
            init_bb = None

    for idx in range(len(frames_from_all_streams)):
           frame = frames_from_all_streams[idx]
           cv2.imshow('Source_ID: '+str(idx), frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        break

    for idx in range(len(frames_from_all_streams)):
        if key == ord(str(idx)):
            selected_src = idx
            ROIs = cv2.selectROIs('ROI Selection..', frames_from_all_streams[selected_src], showCrosshair=True, fromCenter=False)
            for index, roi in enumerate(ROIs):
                if index == 0:
                    init_bb = roi
                    x1, y1, x2, y2 = init_bb
                    x2 = x2 + x1
                    y2 = y2 + y1
                    init_bb = tuple(init_bb)
                    suspect = frames_from_all_streams[selected_src][y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(temp, 'suspect.jpg'), suspect)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frames_from_all_streams[selected_src], init_bb)
                    has_suspect_exit = False
                else:
                    x1, y1, x2, y2 = roi
                    x2 = x2 + x1
                    y2 = y2 + y1
                    img = frames_from_all_streams[idx][y1:y2, x1:x2]
                    img_name = 'feature_' + str(index) + '.jpg'
                    cv2.imwrite(os.path.join(temp, img_name), img)

    frame_count +=1
for idx in range(len(stream_Array)):
    stream_Array[idx].release()
cv2.destroyAllWindows()
