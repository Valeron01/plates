import time
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


import core.utils as utils
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = 416
    video_path = './data/video/test.mp4'

    saved_model_loaded = tf.saved_model.load('./model', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    vid = cv2.VideoCapture(video_path)

    while True:
        return_value, frame = vid.read()

        frame = frame[200:680, 200:968]

        start_time = time.time()
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.

        image_data = np.float32([image_data, image_data])

        
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=10,
            max_total_size=10,
            iou_threshold=0.45,
            score_threshold=0.25
        )

        
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        

        image, c1, c2 = utils.draw_bbox(frame, pred_bbox)


        result = np.asarray(image)

        fps = 1.0 / (time.time() - start_time) * 2
        print("FPS: %.2f" % fps)
        
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("plate", cv2.WINDOW_AUTOSIZE)



        cv2.imshow("result", cv2.resize(result, (960, 540)))
        plate = frame[c1[1]:c2[1], c1[0]:c2[0]]

        if len(plate) > 0:
            plate = plate[:,:,0]
            plate = cv2.adaptiveThreshold(plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,20)

            cv2.imshow("plate", plate)


        if cv2.waitKey(15) & 0xFF == ord('q'): break

        

        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
