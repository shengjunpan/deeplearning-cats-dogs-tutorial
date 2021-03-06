import os
import sys
import re
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

suffix = '1'
if len(sys.argv) > 1:
    suffix = sys.argv[1]
    
limits = None
if len(sys.argv) > 2:
    limits = int(sys.argv[2])

print('Reading mean image, caffe model and its weights')
mean_blob = caffe_pb2.BlobProto()
with open('model_data/input/mean.binaryproto', 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

glob_pattern = 'model_data/caffe_model_{0}/snapshots/caffe_model_{0}_iter_*.caffemodel'.format(suffix)
match_pattern = 'model_data/caffe_model_{0}/snapshots/caffe_model_{0}_iter_([0-9]+).caffemodel'.format(suffix)
models = glob.glob(glob_pattern)
iter_matcher = re.compile(match_pattern)
if models:
    max_iter = None
    for model in models:
        m = iter_matcher.match(model)
        if m:
            iter = int(m.group(1))
            if max_iter is None or iter > max_iter:
                max_iter = iter
    if max_iter is None:
        raise Exception('No model found: ' + match_pattern)
    else:
        model_file = model
        print('model file:', model_file)
else:
    raise Exception('No model found: ' + glob_pattern)

print('Read model architecture and trained model\'s weights')
net = caffe.Net('caffe_models/caffe_model_{0}/caffenet_deploy_{0}.prototxt'.format(suffix),
                model_file,
                caffe.TEST)

print('Define image transformers')
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

print('Making predicitions')

print('Reading image paths')
test_img_paths = list(glob.glob("model_data/input/test1/*jpg"))
output_img_dir = 'model_data/caffe_model_{}/predictions'.format(suffix)
if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

for img_path in test_img_paths[:limits]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    prob = pred_probas.max()
    label = 'cat' if pred_probas.argmax() == 0 else 'dog'
    cv2.rectangle(img, (0,0), (100,20), (255,255,255), cv2.FILLED)
    cv2.putText(img=img,
                text='{}: {:.2f}%'.format(label, prob*100),
                org=(0,15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0,0,255),
                thickness=2)
    output_img_path = os.path.join(output_img_dir, os.path.basename(img_path))
    cv2.imwrite(output_img_path, img)
    print('Saved prediction to', output_img_path)
