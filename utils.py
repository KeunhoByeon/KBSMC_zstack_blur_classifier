import os

import cv2
import torch


def save_state_dict(save_path, epoch, model, optimizer=None, scheduler=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None, }, save_path)


def load_state_dict(state_dict_path, model, optimizer=None, scheduler=None):
    state_dict = torch.load(state_dict_path)

    epoch = state_dict["epoch"]
    model_state = state_dict["model_state"]
    optimizer_state = state_dict["optimizer_state"]
    scheduler_state = state_dict["scheduler_state"]

    model.load_state_dict(model_state)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    return epoch


def resize_and_pad_image(img, target_size=(512, 512), keep_ratio=False, padding=False, interpolation=None):
    # 1) Calculate ratio
    old_size = img.shape[:2]
    if keep_ratio:
        ratio = min(float(target_size[0]) / old_size[0], float(target_size[1]) / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
    else:
        new_size = target_size

    # 2) Resize image
    if interpolation is None:
        interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img.copy(), (new_size[1], new_size[0]), interpolation=interpolation)

    # 3) Pad image
    if padding:
        delta_w = target_size[1] - new_size[1]
        delta_h = target_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        if (isinstance(padding, list) or isinstance(padding, tuple)) and len(padding) == 3:
            value = padding
        else:
            value = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)

    return img
