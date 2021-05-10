from common import *
import tensorflow as tf
import numpy as np


def iou_vector(y_true_f, y_pred_img):
    """
    Liczone dla każdego kanału - w wyniku otrzymuje się wektor zawierający IoU dla każdego elementu batcha (bez uśrednienia)
    """
    y_true = tf.cast(y_true_f, dtype=tf.int32)
    y_pred = tf.argmax(y_pred_img, axis=-1, output_type=tf.int32)
    for i in range(NUMBER_OF_CHANNELS):
        curr_channel = tf.constant(i)
        c_true = tf.equal(y_true, curr_channel)
        c_pred = tf.equal(y_pred, curr_channel)
        I_local = tf.reduce_sum(tf.cast(tf.logical_and(c_true, c_pred), dtype=tf.int32), axis=(1, 2))
        U_local = tf.reduce_sum(tf.cast(tf.logical_or(c_true, c_pred), dtype=tf.int32), axis=(1, 2))
        if i == 0:
            res = (I_local / U_local)
        else:
            res = res + (I_local / U_local)
    return tf.cast(res / tf.constant(NUMBER_OF_CHANNELS, dtype=tf.float64), dtype=tf.float32)


def iou(y_true_f, y_pred_img):
    return tf.reduce_mean(iou_vector(y_true_f, y_pred_img))


def mean_ap(y_true_f, y_pred_img):
    iou_vec = iou_vector(y_true_f, y_pred_img)
    for t in THRESHOLDS:
        threshold = tf.constant(t, dtype=tf.float32)
        if t == THRESHOLDS[0]:
            result = tf.reduce_mean(tf.cast(tf.greater(iou_vec, threshold), dtype=tf.float32))
        else:
            result = result + tf.reduce_mean(tf.cast(tf.greater(iou_vec, t), dtype=tf.float32))
    k = result / tf.constant(len(THRESHOLDS), dtype=tf.float32)
    return k


def iou_binary(y_true_f, y_pred_img):
    """
    Liczone tylko dla czerwonej maski
    """
    y_true = tf.cast(y_true_f, dtype=tf.int32)
    y_pred = tf.argmax(y_pred_img, axis=-1, output_type=tf.int32)
    I = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    U = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) - I
    result = tf.reduce_mean(I / U)
    return result

###############################
# ........................... #
###############################


class UpdatedThresholdMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=None, threshold=0.5, name=None, dtype=None):
        super(UpdatedThresholdMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_probs = tf.nn.softmax(y_pred, axis=-1, name=None)
        y_probs_base = tf.split(y_probs, 2, axis=-1)[1]
        y_pred = tf.where(y_probs_base > self.threshold, 1, 0)

        return super().update_state(y_true, y_pred, sample_weight)


class SteppedMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes=None, thresholds=None, name=None, dtype=None):
        super(SteppedMeanIoU, self).__init__(name=name, dtype=dtype)
        self.ious = [UpdatedThresholdMeanIoU(num_classes=num_classes, threshold=i) for i in thresholds]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for x in self.ious:
            x.update_state(y_true, y_pred, sample_weight)
        return None

    def reset_states(self):
        for x in self.ious:
            x.reset_states()
        return None

    def result(self):
        return tf.add_n([x.result() for x in self.ious])/len(self.ious)
