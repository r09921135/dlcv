# TODO: create shell script for running your GAN model

wget 'https://www.dropbox.com/s/a4e0r2cmv32z7pn/G_29.7.pth?dl=1'
mv G_29.7.pth?dl=1 ./hw2_1/G_29.7.pth
python3 ./hw2_1/test.py --out_path=$1 --model_path=./hw2_1/G_29.7.pth 
