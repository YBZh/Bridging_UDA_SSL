import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits
from torchvision.datasets.utils import download_url
import numpy as np

def process_test_txt(data_list_file, root):
    original_txt = os.path.join(root, 'test_original.txt')
    with open(original_txt) as txt:
        content = txt.readlines()
        txt.close()
    lines = np.array(content)
    new_lines = ['test/' + line for line in lines]
    with open(data_list_file, 'w') as F:
        F.writelines(new_lines)
        F.close()
    os.remove(original_txt)


class VisDA2017(ImageList):
    """`VisDA-2017 <http://ai.bu.edu/visda-2017/assets/attachments/VisDA_2017.pdf>`_ Dataset

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'T'``: training and ``'V'``: validation.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
                aeroplance/
                    *.png
                    ...
            validation/
            image_list/
                train.txt
                validation.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/b25b2b990e8f42e691f0/?dl=1"),
        ("train", "train.tar", "http://csr.bu.edu/ftp/visda17/clf/train.tar"),
        ("validation", "validation.tar", "http://csr.bu.edu/ftp/visda17/clf/validation.tar"),
        ("test", "test.tar", "http://csr.bu.edu/ftp/visda17/clf/test.tar")
    ]

    target_test = ('test_original.txt', 'https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/data/image_list.txt')

    image_list = {
        "Synthetic": "image_list/train.txt",
        "Real": "image_list/validation.txt",
        "test": "image_list/test.txt"
    }
    CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']


    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        if (not os.path.exists(data_list_file)) and task == 'test':
            ## download and pre-process the test.txt;
            download_url(self.target_test[1], root, self.target_test[0])
            process_test_txt(data_list_file, root)


        super(VisDA2017, self).__init__(root, VisDA2017.CLASSES, data_list_file=data_list_file, **kwargs)
