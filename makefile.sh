# with no physics loss added set --phy_scale 0
# crop_size should be same as image size
python train.py --loss_type L1 --phy_scale 0 --FD_kernel 3 --scale_factor 8 --batch_size 32 --crop_size 128 --epochs 200 --seed 0