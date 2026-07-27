"""
Inference class for the ed_check ResNet18 binary classifier.
Expects images as np.ndarray (HWC, RGB, uint8).
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from shared.tcg_config import TCGConfig


class EditionClassifier:
    def __init__(self, model_path, tcg_config: TCGConfig):
        """
        Load a trained ResNet18 checkpoint (as saved by train()) and prepare it for inference.

        Args:
            checkpoint_path: path to the .pt checkpoint containing
                              "model_state_dict" and "classes"
        """
        self.config = tcg_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.target_idx = self.classes.index("first_ed")

        self.model = self._build_model(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    @staticmethod
    def _build_model(num_classes):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        return self.transform(img)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        h, w, _ = frame.shape
        names = list(self.config.edition_areas.keys())
        tensors = []
        crops = []
        for name in names:
            rel_pos = self.config.edition_areas[name]
            y1 = int(h*rel_pos[0][1])
            y2 = int(h*rel_pos[1][1])
            x1 = int(w*rel_pos[0][0])
            x2 = int(w*rel_pos[1][0])
            crop = frame[y1:y2, x1:x2]
            img = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
            tensors.append(self.transform(img))
            crops.append(crop)

        probs = F.softmax(
            self.model(torch.stack(tensors).to(self.device)), dim=1
        ).cpu()

        return {name: float(probs[i, self.target_idx]) for i, name in enumerate(names)}, crops