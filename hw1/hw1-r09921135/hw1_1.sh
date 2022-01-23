wget 'https://www.dropbox.com/s/7h5fbwc4vidfkgk/0.8256.pth.tar?dl=1'
mv 0.8256.pth.tar?dl=1 ./hw1_1/0.8256.pth.tar
python3 ./hw1_1/test.py --test_dir $1 --out_dir $2
