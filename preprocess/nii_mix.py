import os
import SimpleITK as sitk
import nibabel as nib
import csv

subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']   # 
for sub in subs:
    id = os.listdir("/data1/chundanxu/pgan_npy_input/code7.3/mydata_SRPBS_aver/normalize_imgs/test/"+sub+"/test/")
    path_0 = "/data1/chundanxu/pgan_npy_input/code7.3/results_dir/SRPBS/"+sub+"_aver/SRPBS_aver_source_aug_3/pGAN_run_nii/"
    path = "/data1/chundanxu/pgan_npy_input/code7.3/results_dir/SRPBS/"+sub+"_aver/SRPBS_aver_tta_3-2/pGAN_run_nii/"
    new_path = "/data1/chundanxu/pgan_npy_input/code7.3/results_dir/SRPBS/"+sub+"_aver/SRPBS_aver_tta_3-2/mix_nii/"
    os.makedirs(new_path, exist_ok=True)

    for name in id:
        img_0 = nib.load(path_0+name+'.nii.gz').get_fdata()
        img_1 = nib.load(path+name+'_0.3.nii.gz').get_fdata()
        img_2 = nib.load(path+name+'_-0.3.nii.gz').get_fdata()
        img_3 = nib.load(path+name+'_0.6.nii.gz').get_fdata()
        img_4 = nib.load(path+name+'_-0.6.nii.gz').get_fdata()

        new = nib.Nifti1Image(0.2*img_0+0.2*img_1+0.2*img_2+0.2*img_3+0.2*img_4, nib.load(path_0+name+'.nii.gz').affine)
        new.to_filename(new_path+name+'.nii.gz')