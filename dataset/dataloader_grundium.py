import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.dataset_grundium import GRUNDIUM_SLIDE_INFO, GRUNDIUM_SPLIT
from utils import resize_and_pad_image


class GrundiumZStackDataset(Dataset):
    def __init__(self, args, mode: str = "all", return_path: bool = False):
        self.data_dir = args.data
        self.input_size = args.input_size
        self.mode = mode
        self.return_path = return_path

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.samples = []

        cnt_gt, cnt_stain = {0: 0, 1: 0}, {}
        for path, dir, filenames in os.walk(self.data_dir):
            if "Single_layer" in path:
                continue
            for filename in filenames:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                file_path = os.path.join(path, filename)
                slide_full_name = path.split(os.sep)[-1]
                slide_name = "_".join(slide_full_name.split("_")[:-1])
                zstack_index = int(slide_full_name.split("_")[-1])
                patch_coords = filename.strip(ext).split("_patch_")[-1]

                if not (self.mode is None or self.mode.lower() == "all"):
                    if slide_name not in GRUNDIUM_SPLIT[self.mode.lower()]:
                        continue

                slide_info = GRUNDIUM_SLIDE_INFO[slide_name]
                slide_stain = slide_info["stain"]
                slide_base_layer = slide_info["base_layer"]
                slide_layer = zstack_index - slide_base_layer
                slide_offset = slide_info["offset"]
                offset = slide_layer * slide_offset

                gt = 0 if -2 < offset <= 2 else 1

                self.samples.append((file_path, gt))

                # Count
                if slide_stain not in cnt_stain:
                    cnt_stain[slide_stain] = {0: 0, 1: 0}
                cnt_stain[slide_stain][gt] += 1
                cnt_gt[gt] += 1

        print("[{:<5} dataset] Total: {}({}/{})".format(str(self.mode), len(self), cnt_gt[0], cnt_gt[1]), end="\t")
        for stain_name in sorted(cnt_stain.keys()):
            print("{}: {}({}/{})".format(
                stain_name, cnt_stain[stain_name][0] + cnt_stain[stain_name][1],
                cnt_stain[stain_name][0], cnt_stain[stain_name][1]), end="\t")
        print()

    def __getitem__(self, index):
        img_path, gt = self.samples[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        if self.return_path:
            return img_path, img, gt
        return img, gt

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import argparse

    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Data Arguments
    parser.add_argument('--data', default='/mnt/16742a8d-f473-47ba-bf15-144f0e9eb6d7/data/patch_data/KBSMC/zstack/Grundium_SCAN_20240122_anonymous', help='path to dataset')
    parser.add_argument('--input_size', default=512, type=int, help='image input size')
    args = parser.parse_args()

    all_dataset = GrundiumZStackDataset(args)
    train_dataset = GrundiumZStackDataset(args, mode='train')
    val_dataset = GrundiumZStackDataset(args, mode='val')
    test_dataset = GrundiumZStackDataset(args, mode='test')
