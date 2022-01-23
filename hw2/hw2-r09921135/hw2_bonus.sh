# TODO: create shell script for running your improved UDA model

wget 'https://www.dropbox.com/s/j9rjzvs210joyjc/model_0.610.pth?dl=1'
mv model_0.610.pth?dl=1 ./hw2_3/model_0.610.pth
wget 'https://www.dropbox.com/s/d7ynaei23zo7pnq/model_0.879.pth?dl=1'
mv model_0.879.pth?dl=1 ./hw2_3/model_0.879.pth
wget 'https://www.dropbox.com/s/7r4bs4tiofakkx4/model_0.297.pth?dl=1'
mv model_0.297.pth?dl=1 ./hw2_3/model_0.297.pth
python3 ./hw2_3/test.py --data_path=$1 --target=$2 --out_path=$3 --bonus
