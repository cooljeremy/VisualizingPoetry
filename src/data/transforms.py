from typing import Tuple
from torchvision import transforms as T


def build_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    train_tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    eval_tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    return train_tf, eval_tf


