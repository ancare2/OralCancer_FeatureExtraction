import os
import sys

import numpy as np
import cv2

import tensorflow as tf
import albumentations as A

class NucleiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_ids, img_path, batch_size=8, image_size=128, shuffle=True, augment=False):
        self.ids = image_ids
        self.path = img_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

        # AUMENTACIÓN CON ALBUMENTATIONS
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GridDistortion(p=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
                A.Resize(height=self.image_size, width=self.image_size)
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size)
            ])

    # IMAGENES

    def load_image(self, item):
        item_name = os.path.splitext(os.path.basename(item))[0]
        if item_name.endswith("_1"):
            item_name = item_name[:-2]
        full_image_path = os.path.join(self.path, item_name, "images", item_name + ".png")

        image = cv2.imread(full_image_path, 1)
        if image is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        return image

    # MÁSCARAS
    
    def load_masks(self, item):
        item_name = os.path.splitext(os.path.basename(item))[0]
        item_name_clean = item_name[:-2] if item_name.endswith("_1") else item_name
        mask_file_path = os.path.join(self.path, item_name_clean, "masks", item_name_clean + "_1.png")

        #print(f"[DEBUG] Cargando máscara: {mask_file_path}")
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 0).astype(np.uint8)
        return np.expand_dims(mask.astype(np.float32), axis=-1)
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            current_batch_size = len(self.ids) - index * self.batch_size
        else:
            current_batch_size = self.batch_size

        batch = self.ids[index * self.batch_size: index * self.batch_size + current_batch_size]
        images, masks = [], []

        for item in batch:
            img = self.load_image(item)
            mask = self.load_masks(item)

            if self.augment:
                mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                augmented = self.transform(image=img, mask=mask_uint8)
                img = augmented['image']
                mask_aug = augmented['mask']
                mask = np.expand_dims((mask_aug > 127).astype(np.float32), axis=-1)

            images.append(img)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)

        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.ids)

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))