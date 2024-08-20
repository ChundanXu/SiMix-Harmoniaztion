import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np

uih_path = '/data1/chundanxu/pgan_npy_input/code/mydata_CG/normalize_imgs/test/'
syn_path = '/data1/chundanxu/pgan_npy_input/code7.3/mydata_SRPBS_aver_source_aug_3_all/results_dir/CG/ge_to_average/SRPBS_aver_source_aug_3/pGAN_run_nii/'
new_path = '/data1/chundanxu/pgan_npy_input/code7.3/mydata_SRPBS_aver_source_aug_3_all/results_dir/CG/ge_to_average/SRPBS_aver_source_aug_3/mix/test/'
id = os.listdir(uih_path)
alphas = ['0.1', '-0.1', '0.2', '-0.2']

for name in id:
    uih_file = os.listdir(os.path.join(uih_path, name))
    uih_file.sort()
    header = nib.load(os.path.join(uih_path, name, uih_file[0]))
    uih = nib.load(os.path.join(uih_path, name, uih_file[0])).get_fdata()
    print(uih_file[0], uih.max())

    syn = nib.load(syn_path+name+'.nii.gz').get_fdata()
    nor = syn.max()
    print(syn.max())
    syn = syn / nor
    print(syn.max())

    for alpha in alphas:
        alpha_path = new_path+name+"_"+str(alpha)
        os.makedirs(alpha_path, exist_ok=True)
        
        alpha = float(alpha)
        print("alpha = ", alpha)
        syn=(1-alpha)*uih+alpha*syn
        
        for x in range(header.shape[0]):
            for y in range(header.shape[1]):
                for z in range(header.shape[2]):
                    if syn[x,y,z]<0:
                        syn[x,y,z]=0
                    elif syn[x,y,z]>1:
                        syn[x,y,z]=1
        print(syn.min(), syn.max())

        new = nib.Nifti1Image(syn, header.affine)
        new.to_filename(os.path.join(new_path, alpha_path, uih_file[0]))

        cmd = "scp -r " + os.path.join(uih_path, name, uih_file[1]) + " " + os.path.join(new_path, alpha_path, uih_file[1])
        print(cmd)
        os.system(cmd)



""" path = '/data02/chundanxu/data_all_pgan/pgan_npy_input_mix/mydata/normalize_imgs/train/'
new_path = '/data02/chundanxu/data_all_pgan/pgan_npy_input_mix/mydata/normalize_imgs/base+source_aug/train/'
subs = os.listdir(path)

for sub in subs:
    files = os.listdir(path+sub)
    for file in files:
        if "ge" in file:
            ge_file = file
            ge = nib.load(os.path.join(path, sub, ge_file)).get_fdata()
        elif "philips" in file:
            philips_file = file
            philips = nib.load(os.path.join(path, sub, philips_file)).get_fdata()
            header = nib.load(os.path.join(path, sub, philips_file))
    for try_id in range(1,21):
        os.makedirs(new_path+sub+"_try_"+str(try_id), exist_ok=True)
        cmd = "scp -r " + os.path.join(path, sub, ge_file) + " " + new_path+sub+"_try_"+str(try_id)+"/"+ge_file
        print(cmd)
        os.system(cmd)

        for i in range(160):
            alpha = np.random.uniform(-0.2,0.2,1)
            # print(alpha)
            philips[:,:,i]=alpha * (ge[:,:,i]) + (1-alpha) * (philips[:,:,i])
        print(philips.min(), philips.max())

        for x in range(168):
            for y in range(256):
                for z in range(160):
                    if philips[x,y,z]<0:
                        philips[x,y,z]=0
                    elif philips[x,y,z]>1:
                        philips[x,y,z]=1
        print(philips.min(), philips.max())

        new = nib.Nifti1Image(philips, header.affine)
        new.to_filename(new_path+sub+"_try_"+str(try_id)+"/"+philips_file) """
