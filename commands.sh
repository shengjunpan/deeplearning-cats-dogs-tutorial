mkdir -p model_data/input

unzip ~/Downloads/train.zip -d model_data/input
unzip ~/Downloads/test1.zip -d model_data/input

python3 code/create_lmdb.py 

compute_image_mean -backend=lmdb model_data/input/train_lmdb model_data/input/mean.binaryproto

python3 $CAFFE_HOME/python/draw_net.py \
	caffe_models/caffe_model_1/caffenet_train_val_1.prototxt \
	caffe_models/caffe_model_1/caffenet_train_val_1.png

mkdir -p model_data/caffe_model_1/snapshots

caffe train \
      --solver caffe_models/caffe_model_1/solver_1.prototxt \
      2>&1 | tee model_data/caffe_model_1/model_1_train.log

python3 code/plot_learning_curve.py \
	--rankdir TB \
	model_data/caffe_model_1/model_1_train.log \
        caffe_models/caffe_model_1/caffe_model_1_learning_curve.png

python3 code/make_predictions.py 1 5

###############################################
mkdir -p model_data/caffe_model_2/snapshots

wget -P model_data/caffe_model_2/ http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

caffe train \
      --solver caffe_models/caffe_model_2/solver_2.prototxt \
      --weights model_data/caffe_model_2/bvlc_reference_caffenet.caffemodel \
      2>&1 | tee model_data/caffe_model_2/model_2_train.log

python3 code/plot_learning_curve.py \
	--rankdir TB \
        model_data/caffe_model_2/model_2_train.log \
        caffe_models/caffe_model_2/caffe_model_2_learning_curve.png

python3 code/make_predictions.py 2 5
