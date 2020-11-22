import tensorflow as tf
import sys
from draw import select_object, draw_image
import cv2
from models import Yolov4
from os.path import join

def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def go_outside(obj_coords, target_coords):

    iou = bb_intersection_over_union(obj_coords, target_coords)
    if iou >= 0.2:
        return True
    else:
        return False


def go_inside(prev_coords, target_coords):

    iou1 = bb_intersection_over_union(prev_coords, target_coords)
    if iou1 >= 0.4:
        return True
    else:
        return False


def same_id(previous_id, current_id):
    return previous_id==current_id


if __name__ == "__main__":
    output_format = 'mp4'
    video_name = '2.avi'
    output_name = 'save_data/out_' + video_name[0:-3] + output_format

    is_first_frame = True
    initialize_door_by_yourself = False
    num_frame = 0
    counter_detection = 0
    person_id = 0
    prev_id = 0
    prev_coords = None
    people_actions = dict()
    person_inside = dict()
    person_outside = dict()

    people_actions['in'] = 0
    people_actions['out'] = 0

    check_gpu()
    model = Yolov4(weight_path='yolo4_config/yolov4.weights',
                   class_name_path='yolo4_config/coco_classes.txt')

    videofile = cv2.VideoCapture(video_name)
    fps = videofile.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video = cv2.VideoWriter(output_name, fourcc, fps, (640, 480))
    rr, first_frame = videofile.read()

    if initialize_door_by_yourself:
        door_array = select_object(first_frame)[0]
    else:
        door_array = [361, 20, 507, 352]

    while rr:
        ret, frame = videofile.read()
        if not ret:
            break
        detections = model.predict_img(frame, random_color=False, plot_img=False)
        for detection in detections.values:
            if detection[4] == 'person':
                counter_detection += 1

                coordinates = detection[0:4]
                if prev_coords is None:
                    prev_coords = coordinates.copy()
                    person_id = 1

                iou = bb_intersection_over_union(coordinates, prev_coords)

                if iou <= 0.4 and (num_frame - counter_detection) > 10:
                    person_id += 1

                if not same_id(prev_id, person_id):
                    if (num_frame - counter_detection < 5) and go_outside(prev_coords, door_array):
                        people_actions['out'] += 1
                        person_outside['person_id'] = person_id

                counter_detection = num_frame
                prev_coords = coordinates.copy()

        if (num_frame - counter_detection > 5) and go_inside(prev_coords, door_array)\
                and person_id not in person_inside.values():
            people_actions['in'] += 1
            person_inside['person_id'] = person_id

        output_frame = draw_image(frame, people_actions)
        num_frame += 1
        prev_id = person_id

        output_video.write(output_frame)

    output_video.release()
    cv2.destroyAllWindows()