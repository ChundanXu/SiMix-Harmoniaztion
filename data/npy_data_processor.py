import os
import torch
import numpy as np
from torch.utils.data import Dataset


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        for model in os.listdir(os.path.join(path, file)):
            if os.path.splitext(model)[1] == '.gz':
                file_path = os.path.join(path, file, model)
                tmp_list.append(file_path)
    return tmp_list


class ImageDataset(Dataset):
    def __init__(self, opt):
        target_dir = os.path.join(opt.dataroot, opt.phase)
        patients_name_list = os.listdir(target_dir)
        self.model_y = opt.model_y
        self.model_x_list = opt.model_xs
        if isinstance(self.model_x_list, str):
            self.model_x_list = [opt.model_xs]

        self.all_image_list = []

        for patient in patients_name_list:
            patient_model_y = os.path.join(target_dir, patient, self.model_y)
            for slicer in os.listdir(patient_model_y):
                self.all_image_list.append(os.path.join(target_dir, patient, self.model_y, slicer))

        # check files count across models
        data_x_nii_files_count_list = np.array(
            [len(os.listdir(os.path.join(target_dir, patient))) for patient in patients_name_list])
        print(data_x_nii_files_count_list)
        print(f'NPY files\'s count checked, {len(data_x_nii_files_count_list)} files in each model.')

    def __getitem__(self, index):
        name = self.all_image_list[index]
        data_return = {}

        nmupy_A = np.array([np.load(name.replace(self.model_y, model)) for model in self.model_x_list]).astype(np.float32)
        nmupy_B = np.load(name).astype(np.float32)

        data_return['A'] = torch.from_numpy(nmupy_A)
        data_return['B'] = torch.from_numpy(nmupy_B).unsqueeze(0)
        data_return['A_paths'] = name
        data_return['B_paths'] = name

        return data_return

    def __len__(self):
        return len(self.all_image_list)
