import SimpleITK as sitk
import os
import numpy as np
from tqdm import trange
from options.test_options import TestOptions
import nibabel as nib


def get_npy_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.npy':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list

def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        for model in os.listdir(os.path.join(path, file)):
            if os.path.splitext(model)[1] == '.gz':
                file_path = os.path.join(path, file, model)
                tmp_list.append(file_path)
    return tmp_list

if __name__ == '__main__':
    opt = TestOptions().parse()
    nii_path = os.path.join(opt.dataroot, 'test')
    save_path = os.path.join(opt.results_dir, opt.name + '_nii')
    os.makedirs(save_path, exist_ok=True)
    patients_name_list = os.listdir(nii_path)
    patients_path_list = [os.path.join(nii_path, i) for i in patients_name_list]
    # patients_model_y_list = [os.path.join(i, opt.model_y) for i in patients_path_list]
    patients_name_list.sort()
    patients_path_list.sort()
    # patients_model_y_list.sort()
    print(patients_path_list)

    for i in range(len(patients_path_list)):
        files = os.listdir(patients_path_list[i])
        sitk_img = sitk.ReadImage(os.path.join(patients_path_list[i], files[0]))
        img_arr = sitk.GetArrayFromImage(sitk_img)
        img_list = [i for i in get_npy_listdir(os.path.join(opt.results_dir, opt.name, patients_name_list[i]))]
        img_list.sort()
        new_img = np.zeros_like(img_arr)
        for j in trange(len(img_list)):
            image = np.load(img_list[j])
            new_img[j, :, :] = image
        # print(new_img.min(), new_img.max(), new_img.shape)
        for x in range(new_img.shape[0]):
            for y in range(new_img.shape[1]):
                for z in range(new_img.shape[2]):
                    if new_img[x,y,z] < 0:
                        new_img[x,y,z] = 0
                    elif new_img[x,y,z] > 1:
                        new_img[x,y,z] = 1
                    # new_img[x,y,z] = new_img[x,y,z] *255
        # print(new_img.min(), new_img.max())

        new_img = sitk.GetImageFromArray(new_img)
        new_img.SetDirection(sitk_img.GetDirection())
        new_img.SetSpacing(sitk_img.GetSpacing())
        new_img.SetOrigin(sitk_img.GetOrigin())
        sitk.WriteImage(new_img, os.path.join(save_path, patients_name_list[i] + '.nii.gz'))

        """ sitk_img = nib.load(os.path.join(patients_path_list[i], files[0]))
        img_list = [i for i in get_npy_listdir(os.path.join(opt.results_dir, opt.name, patients_name_list[i]))]
        img_list.sort()
        first_loop = True
        for j in trange(len(img_list)):
            temp_concat = np.expand_dims(np.load(img_list[j])[:,:,0],axis=2)
            if first_loop:
                    concat = temp_concat
                    first_loop = False
            else:
                    concat = np.concatenate((concat,temp_concat),axis=2)
            
        # nii = nib.Nifti1Image(concat.transpose((1,0,2)), sitk_img.affine)
        nii = nib.Nifti1Image(concat, sitk_img.affine)
        name = os.path.join(save_path, patients_name_list[i] + '.nii.gz')
        nii.to_filename(name) """
