rm -rf log/*
nohup python -u train_keras_bert.py >> log/train.log 2>&1 &