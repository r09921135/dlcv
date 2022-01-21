# Download pretrained weights
[ -f model_pretrain.pth ] && echo "Model Pretrain exists" || wget -O model_pretrain.pth 'https://www.dropbox.com/s/bjg0mmhqdcr68lz/tade_model_checkpoint.pth?dl=1'

# Training
python3 train.py -c config.json -r model_pretrain.pth -f $1

# Test-time aggregation training 
python3 test_training_food.py -c test_time_config.json -r saved/models/Food_LT_ResNeXt50_SSKD/model_best.pth -f $1
