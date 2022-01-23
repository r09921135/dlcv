wget 'https://www.dropbox.com/s/3oxylon1a8kvkkp/0.7029.pth?dl=1'
mv 0.7029.pth?dl=1 0.7029.pth
python3 ./hw1_2/test.py --model_dir ./0.7029.pth --test_dir $1 --out_dir $2
