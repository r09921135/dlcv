# TODO: create shell script for running your ViT testing code

wget 'https://www.dropbox.com/s/0m70ryaletwcf4f/0.9500.pth.tar?dl=1'
mv 0.9500.pth.tar?dl=1 ./hw3_1/0.9500.pth.tar
python3 ./hw3_1/test.py --test_dir $1 --out_dir $2 --model_dir ./hw3_1/0.9500.pth.tar 
