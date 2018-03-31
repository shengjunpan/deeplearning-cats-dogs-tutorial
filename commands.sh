mkdir -p model_data/input

unzip ~/Downloads/train.zip -d model_data/input
unzip ~/Downloads/test1.zip -d model_data/input

python code/create_lmdb.py 
compute_image_mean -backend=lmdb model_data/input/train_lmdb model_data/input/mean.binaryproto

python ~/Downloads/caffe/python/draw_net.py \
       --phase TRAIN \
       --rankdir TB \
       caffe_models/caffe_model_1/caffenet_train_val_1.prototxt \
       caffe_models/caffe_model_1/caffenet_train_val_1.png

mkdir -p model_data/caffe_model_1/snapshots
caffe train \
      --solver caffe_models/caffe_model_1/solver_1.prototxt \
      2>&1 | tee model_data/caffe_model_1/model_1_train.log
python code/plot_learning_curve.py \
       model_data/caffe_model_1/model_1_train.log \
       caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
python code/make_predictions_1.py 

mkdir -p model_data/caffe_model_2/snapshots
caffe train \
      --solver caffe_models/caffe_model_2/solver_2.prototxt \
      --weights model_data/bvlc_reference_caffenet.caffemodel \
      2>&1 | tee model_data/caffe_model_2/model_2_train.log
python code/plot_learning_curve.py \
       model_data/caffe_model_2/model_2_train.log \
       caffe_models/caffe_model_2/caffe_model_2_learning_curve.png
