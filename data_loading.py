import glob
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy
from PIL import Image


class VData(Dataset):
    def __init__(self, parovi, transforms):
        self.parovi = parovi
        self.transforms = transforms

    def custom_transform(self, og, mask, weights):
        height = random.randrange(700, 1024)
        width = random.randrange(300, 512)
        i, j, h, w = transforms.RandomCrop.get_params(og, (height, width))
        degrees, translate, scale_ranges, shears = transforms.RandomAffine.get_params((0, 0), (-0.15, 0.15), None, None,
                                                                                      (1024, 512))

        og = F.affine(og, 0, translate, scale_ranges, shears)
        og = F.crop(og, i, j, h, w)
        mask = F.affine(mask, 0, translate, scale_ranges, shears)
        mask = F.crop(mask, i, j, h, w)
        weights = F.affine(weights, 0, translate, scale_ranges, shears)
        weights = F.crop(weights, i, j, h, w)

        if random.random() > 0.5:
            og = F.hflip(og)
            mask = F.hflip(mask)
            weights = F.hflip(weights)

        return og, mask, weights

    def __getitem__(self, index):

        mask_img_rgb = Image.open(self.parovi[index][1])

        if index % 2 == 0:
            self.transforms(mask_img_rgb)

        mask_img_grayscale = mask_img_rgb.convert('L')
        mask_img = numpy.asarray(mask_img_grayscale)

        mask_img_normalized = mask_img.copy()
        numpy.place(mask_img_normalized, mask_img_normalized == 64, 1)
        numpy.place(mask_img_normalized, mask_img_normalized == 128, 2)
        numpy.place(mask_img_normalized, mask_img_normalized == 192, 3)

        og_img_bgr = cv2.imread(self.parovi[index][0])
        og_img_gray = cv2.cvtColor(og_img_bgr, cv2.COLOR_BGR2GRAY)

        og_img_gray_max = numpy.max(og_img_gray[:])
        og_img_gray_min = numpy.min(og_img_gray[:])
        og_img_normalized = (og_img_gray - og_img_gray_min) / (og_img_gray_max - og_img_gray_min)
        og_img_normalized = numpy.expand_dims(og_img_normalized, axis=0)

        mask_img_normalized = mask_img_normalized.astype(numpy.float32)
        og_img_normalized = og_img_normalized.astype(numpy.float32)
        return og_img_normalized, mask_img_normalized

    def __len__(self):
        return len(self.parovi)


def get_datasets():
    train_masks = glob.glob("/content/drive/My Drive/pacijenti_dataset/train/*/mask/*")
    train_ogs = glob.glob("/content/drive/My Drive/pacijenti_dataset/train/*/original/*")
    val_masks = glob.glob("/content/drive/My Drive/pacijenti_dataset/val/*/mask/*")
    val_ogs = glob.glob("/content/drive/My Drive/pacijenti_dataset/val/*/original/*")
    test_masks = glob.glob("/content/drive/My Drive/pacijenti_dataset/test/*/mask/*")
    test_ogs = glob.glob("/content/drive/My Drive/pacijenti_dataset/test/*/original/*")

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for mask_path in train_masks:
        mask_file = Path(mask_path).name

        for original_path in train_ogs:
            original_file = Path(original_path).name
            if mask_file == original_file:
                train_dataset.append([original_path, mask_path])

    for mask_path in val_masks:
        mask_file = Path(mask_path).name

        for original_path in val_ogs:
            original_file = Path(original_path).name
            if mask_file == original_file:
                val_dataset.append([original_path, mask_path])

    for mask_path in test_masks:
        mask_file = Path(mask_path).name

        for original_path in test_ogs:
            original_file = Path(original_path).name
            if mask_file == original_file:
                test_dataset.append([original_path, mask_path])

    return VData(train_dataset), VData(val_dataset), VData(test_dataset)
