caffe train \
      --solver caffe_models/caffe_model_1/solver_1.prototxt \
      2>&1 \
    | tee model_data/caffe_model_1/model_1_train.log
