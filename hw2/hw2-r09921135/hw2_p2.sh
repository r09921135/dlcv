# TODO: create shell script for running your ACGAN model

wget 'https://www.dropbox.com/s/x023274ik57mqt7/netG_0.903.pth?dl=1'
mv netG_0.903.pth?dl=1 ./hw2_2/netG_0.903.pth
python3 ./hw2_2/test.py --out_path=$1 --model_path=./hw2_2/netG_0.903.pth
