import os
import argparse
import numpy as np
import tqdm
import SimpleITK as sitk
import nibabel as nib

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./mydata/ori_data', help='data base path')
    parser.add_argument('--phase', type=str, default='train', help='data type, train or test')
    parser.add_argument('--target_shape', type=int, default=256, help='target size')
    parser.add_argument('--source_modality', type=str, action='append',default=['COI', 'HKH', 'KUS', 'KUT'], help='source scanner modality')
    parser.add_argument('--target_modality', type=str, default='ATV', help='target scanner modality')
    parser.add_argument('--LowerBound', type=int, default=50, help='neckcut LowerBound')
    parser.add_argument('--UpperBound', type=int, default=210, help='neckcut UpperBound')

    args = parser.parse_args()
    return args

def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        for model in os.listdir(os.path.join(path, file)):
            if os.path.splitext(model)[1] == '.gz':
                file_path = os.path.join(path, file, model)
                tmp_list.append(file_path)
    return tmp_list

def list_sort(nii_list, source_modality, target_modality):
    if id(source_modality) < id(target_modality):
        return nii_list.sort()
    else:
        return nii_list.sort(reverse=True)

def resize_img(ori_path, ori_dir, target_shape: int):
    resampler = sitk.ResampleImageFilter()
    resize_dir = 'resize_img'
    new_path = ori_path.replace(ori_dir, resize_dir)
    if not os.path.exists(new_path):
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        ori_img = sitk.ReadImage(ori_path)
        origin_size = ori_img.GetSize()
        origin_spacing = ori_img.GetSpacing()

        remainder = [index % 4 for index in origin_size]
        # print(origin_size, remainder)
        if remainder[0] !=0:
            new_size = [origin_size[0] + 4 - int(remainder[0]), target_shape, target_shape]
        else:
            new_size = [origin_size[0], target_shape, target_shape]
        # print(origin_size, new_size)
        new_size = np.array(new_size, dtype=float)
        factor = origin_size / new_size
        newSpacing = origin_spacing * factor
        new_size = new_size.astype(np.int)

        resampler.SetReferenceImage(ori_img)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        new_img = resampler.Execute(ori_img)
        sitk.WriteImage(new_img, new_path)
    return new_path, resize_dir

def register_imgs(resize_path, resize_dir, source_modality, target_modality):
    register_dir = 'register_imgs'
    new_path = resize_path.replace(resize_dir, register_dir)
    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    # print(new_path)
    ants_sh = "antsRegistration"
    additional_sh_1 = " --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 --output ["
    additional_sh_2 = "] --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [0.005,0.995] --initial-moving-transform ["
    additional_sh_3 = "] --transform Rigid[0.1] --metric MI["
    additional_sh_4 = "] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox"

    dir, file = os.path.split(resize_path)
    if not os.path.exists(new_path):
        for i in range(len(source_modality)):
            if source_modality[i] in file:
                file = file.replace(source_modality[i], target_modality)
                target_img_file = os.path.join(dir, file)
                cmd = ants_sh + additional_sh_1+ os.path.split(new_path)[0] + "," + new_path + additional_sh_2 + target_img_file + "," + resize_path + ",1" + additional_sh_3+ target_img_file + "," + resize_path + ",1,32,Regular,0.25" + additional_sh_4
                print(cmd)
                os.system(cmd)
        if target_modality in file:
            cmd = "scp -r " + resize_path + ' ' + new_path
            # print(cmd)
            os.system(cmd)
    return new_path, register_dir

def neckcut_imgs(register_path, register_dir, LowerBound: int, UpperBound: int):
    neckcut_dir = 'neckcut_imgs'
    new_path = register_path.replace(register_dir, neckcut_dir)
    if not os.path.exists(new_path):
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        niigz_file = register_path
        img = sitk.ReadImage(niigz_file)
        neckcut_img = img[:,:,LowerBound:UpperBound]
        sitk.WriteImage(neckcut_img, new_path)   
    return new_path, neckcut_dir

def mask_imgs(neckcut_path, neckcut_dir):
    mask_dir = 'mask_imgs'
    new_path = neckcut_path.replace(neckcut_dir, mask_dir)
    if not os.path.exists(new_path):
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        niigz_file = neckcut_path
        mask_path = new_path
        cmd = 'bet2 ' + niigz_file + ' ' + mask_path + ' -f 0.6 -m'
        # print(cmd)
        os.system(cmd)   
    return new_path, mask_dir


def normalize_imgs(neckcut_path, neckcut_dir):
    normalize_dir = 'normalize_imgs'
    new_path = neckcut_path.replace(neckcut_dir, normalize_dir)
    niigz_file = neckcut_path
    normed_path = new_path
    if not os.path.exists(new_path):
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        img_nii = nib.load(niigz_file)
        img = img_nii.get_fdata()
        assert (img>=0).all()
        if (img>1).any():
            nor = img.max()
            normed_img = img / nor
        else:
            normed_img = img
        saved_img = nib.Nifti1Image(normed_img, img_nii.affine, img_nii.header)
        saved_img.to_filename(normed_path)   
    return new_path, normalize_dir

def nii2npy(normalize_path, normalize_dir, source_modality, target_modality, phase):
    nii2npy_dir = 'nii2npy'
    new_path = normalize_path.replace(normalize_dir, nii2npy_dir)
    dir, file = os.path.split(new_path)
    for i in range(len(source_modality)):
        if source_modality[i] in file:
            new_path = os.path.join(dir, source_modality[i])
    if target_modality in file:
        new_path = os.path.join(dir, target_modality)
    # print(new_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path, exist_ok=True)
        sitk_img = sitk.ReadImage(normalize_path)
        img_arr = sitk.GetArrayFromImage(sitk_img)
    for i in range(img_arr.shape[0]):
        np.save(os.path.join(new_path, str(i).rjust(3, '0') + '.npy'), img_arr[i, :, :])

if __name__ == '__main__':
    args = initialize()
    base_path = args.dataroot
    ori_dir = 'ori_data'
    nii_list = get_listdir(os.path.join(base_path, args.phase))
    list_sort(nii_list, source_modality=args.source_modality, target_modality=args.target_modality)
    print(nii_list)
    for i in tqdm.tqdm(nii_list):
        resize_path, resize_dir = resize_img(i, ori_dir, target_shape=args.target_shape)
        register_path, register_dir = register_imgs(resize_path, resize_dir, source_modality=args.source_modality, target_modality=args.target_modality)
        neckcut_path, neckcut_dir = neckcut_imgs(register_path, register_dir, LowerBound=args.LowerBound, UpperBound=args.UpperBound)
        mask_path, mask_dir = mask_imgs(neckcut_path, neckcut_dir)
        normalize_path, normalize_dir = normalize_imgs(neckcut_path, neckcut_dir)
        nii2npy(normalize_path, normalize_dir, source_modality=args.source_modality, target_modality=args.target_modality, phase=args.phase)