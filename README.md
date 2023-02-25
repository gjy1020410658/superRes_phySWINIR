# superRes_phySWINIR
The run_model_128_noP.py is the file without physics loss added. 
The run_model_128.py is the file with physics loss added. 

A code to run. 
python run_model_128_noP.py 
--loss_type $L1 or $L2 denotes data loss type 
--phy_scale $number denotes scaling parameter before physics loss 
--FD_kernel $3 or $5 denotes finite difference kernal being used
--scale factor $int
--batch_size 
--crop_size should be same as HR dimension.  
