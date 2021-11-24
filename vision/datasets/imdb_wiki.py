import pathlib
import cv2
import numpy as np
from .utils import load_data

class IMDBWikiDataset:
    def __init__(self, root, transform=None, target_transform=None, split='train'):
        self.root = pathlib.Path(root) / split
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths = self._load_data()
        self.class_names = ['BACKGROUND', 'face']
        self.num_gender_classes = 2

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = image_path.replace('/images/', '/labels/').replace('.jpg', '.txt')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        boxes, labels, genders = self._load_label(label_path, img_h, img_w)

        if self.transform:
            image, boxes, labels, genders = self.transform(image, boxes, labels, genders)
        if self.target_transform:
            boxes, labels, genders = self.target_transform(boxes, labels, genders)
        return image, boxes, labels, genders
    
    def _load_data(self):
        image_dir = self.root / 'images'
        image_paths = [str(p) for p in image_dir.glob('*.jpg')]
        return image_paths

    def _load_label(self, label_path, img_h, img_w):
        with open(label_path) as f:
            bboxes = []
            labels = []
            genders = []
            for line in f:
                row = list(map(float, line.strip().split(' ')))
                class_id, cx, cy, w, h, gender = row
                x1 = int((cx - w/2) * img_w)
                y1 = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                bboxes.append((x1, y1, x2, y2))
                labels.append(int(class_id))
                genders.append(int(gender))
            return (
                np.array(bboxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(genders, dtype=np.int64)
            )

    def __len__(self):
        return len(self.image_paths)