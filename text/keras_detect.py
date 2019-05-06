"""
YOLO_v3 Model Defined in Keras.
Reference: https://github.com/qqwweee/keras-yolo3.git
"""
from config import kerasTextModel,IMGSIZE,keras_anchors,class_names,GPU,GPUID
from text.keras_yolo3 import yolo_text,box_layer,K,Model

from apphelper.image import resize_im,letterbox_image
from PIL import Image
import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()##解决web.py 相关报错问题

anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)

num_classes = len(class_names)
textModel = yolo_text(num_classes,anchors)
textModel.load_weights(kerasTextModel)

# sess = K.get_session()
# image_shape = K.placeholder(shape=(2, ))##图像原尺寸:h,w
image_shape = tf.keras.layers.Input(shape=(2, ))
# input_shape = K.placeholder(shape=(2, ))##图像resize尺寸:h,w
# Added by Bo Zhou
# yolo3 dir is copied directly from the referred repo
from yolo3.model import yolo_eval
# box_score = box_layer([*textModel.output,image_shape,input_shape],anchors, num_classes)
yolo_eval = tf.keras.layers.Lambda(yolo_eval,
                                   arguments={"anchors": anchors,
                                              "num_classes": num_classes,
                                              "image_shape": image_shape,
                                              "score_threshold": .1,
                                              "iou_threshold": .8})
_boxes, _scores, _ = yolo_eval(textModel.output)

# wrapped end-to-end model
keras_model = Model([textModel.input, image_shape], [_boxes, _scores])
# compile model to setup tfpark preqs
keras_model.compile(optimizer="rmsprop")
from zoo import init_nncontext
from zoo.tfpark import KerasModel
_ = init_nncontext()
keras_model = KerasModel(keras_model)

def text_detect(img,prob = 0.05):
    im    = Image.fromarray(img)
    # scale = IMGSIZE[0]
    w,h   = im.size
    # w_,h_ = resize_im(w,h, scale=scale, max_scale=2048)##短边固定为608,长边max_scale<4000
    boxed_image,f = letterbox_image(im, IMGSIZE)
    # boxed_image = im.resize((w_,h_), Image.BICUBIC)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    imgShape   = np.array([[h,w]])
    inputShape = np.array([IMGSIZE])


    global graph
    with graph.as_default():
         ##定义 graph变量 解决web.py 相关报错问题
         """
         pred = textModel.predict_on_batch([image_data,imgShape,inputShape])
         box,scores = pred[:,:4],pred[:,-1]
         
         """
         box, scores = keras_model.predict([image_data, imgShape],
                                           distributed=True)

    keep = np.where(scores>prob)
    box[:, 0:4][box[:, 0:4]<0] = 0
    box[:, 0][box[:, 0]>=w] = w-1
    box[:, 1][box[:, 1]>=h] = h-1
    box[:, 2][box[:, 2]>=w] = w-1
    box[:, 3][box[:, 3]>=h] = h-1
    box = box[keep[0]]
    scores = scores[keep[0]]
    # Added by Bo ZHOU
    # If we use yolo_eval from referred repo, we need to exchange the
    # column and row value of detected boxes to keep the same detection
    # orientation and hence the subsequent process would be right
    _box = np.zeros(box.shape)
    _box[..., 0] = box[..., 1]
    _box[..., 1] = box[..., 0]
    _box[..., 2] = box[..., 3]
    _box[..., 3] = box[..., 2]
    return _box,scores

