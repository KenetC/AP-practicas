import os 
import skimage.io as io
import torch 
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pathlib
import re
import numpy as np 

## Ordena strings en orden natural, sirve para ordenar paths
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

## La clase Dataset requiere definir 2 funciones:
## __len__ debe devolver la cantidad de elementos presentes en el dataset

## __getitem__ recibe un indice y debe devolver el elemento asociados a ese indice
## Para ello tenemos los paths a las imágenes guardadas en una lista ordenada alfabeticamente

## También puede recibir una lista de transformaciones, las cuales se definirán posteriormente.
## Éstas se aplican al leer cada dato, aquí se pueden utilizar transformaciones aleatorias para aumentación online
## Las transformaciones se cargan al instanciar el dataset.
class OurDataset(Dataset):
    def __init__(self, PATH, transform=None):

        self.img_path = os.path.join(PATH, 'Images')
        self.label_path = os.path.join(PATH, 'Labels')
        self.transform = transform

        data_root = pathlib.Path(self.img_path)
        all_files = list(data_root.glob('*.png'))
        all_files = [str(path) for path in all_files]
        all_files.sort(key = natural_key)

        self.images = all_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        image = io.imread(img_name, as_gray = True).astype('float')
        image = np.expand_dims(image, axis=2)

        label = img_name.replace(self.img_path, self.label_path)
        label = io.imread(label, as_gray = True)
        label = label.astype(bool).astype(float)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

## Transformaciones a utilizar

## Rescale: sirve para que todos los datos tengan el mismo tamaño,
## reescalandolos a un ancho y alto establecido
class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        lab = transform.resize(label, (new_h, new_w))

        return {'image': img, 'label': lab}
## ToTensor: esta transformación es esencial, ya que convierte los arreglos de numpy a tensores de PyTorch
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long()}
