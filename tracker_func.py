import numpy as np

from sort import Sort
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from deepsort import preprocessing
from yolo3.utils import convert_boxes


def sort_image(sort_class: Sort, out_boxes, out_scores, out_classes):
    dets = []

    for i in range(0, len(out_boxes)):
        dets.append([out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2], out_scores[i], out_classes[i]])

    dets = np.array(dets)
    # update
    trackers = sort_class.update(dets)

    out_boxes = []
    out_scores = []
    out_classes = []
    object_id = []
    # d [x1,y1,x2,y2,object_id,score,type]
    for d in trackers:
        out_boxes.append(list([d[1], d[0], d[3], d[2]]))
        object_id.append(int(d[4]))
        out_scores.append(float(d[5]))
        out_classes.append(int(d[6]))

    return np.array(out_boxes), np.array(out_scores), np.array(out_classes), np.array(object_id)


def deepsort_image(deepsort_class: Tracker, encoder, frame, out_boxes, out_scores, out_classes,
                   nms_max_overlap=1.0):

    converted_boxes = convert_boxes(out_boxes)
    features = encoder(frame, converted_boxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, out_scores, out_classes, features)]

    # run non-maxima suppresion
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    deepsort_class.predict()
    deepsort_class.update(detections)

    num_trackers = len(deepsort_class.tracks)
    out_boxes = []
    out_classes = []
    out_scores = []
    object_id = []
    # d [x1,y1,x2,y2,object_id,score,type]
    for index, track in enumerate(deepsort_class.tracks):
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        out_boxes.append(track.to_tlbr())
        out_classes.append(int(track.get_class()))
        out_scores.append(float(track.get_score()))
        object_id.append(int(track.track_id))

    return np.array(out_boxes), np.array(out_scores), np.array(out_classes), np.array(object_id)
