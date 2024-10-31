import os
import mindspore
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from download import download
from config import MnistConfig

def datapipe(dataset, batch_size):
    """ process input dataset with pipeline """
    image_transforms = [
        vision.Rescale((1 / 255), 0),  # picture points rescaling
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transfroms = transforms.TypeCast(mindspore.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transfroms, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

def get_dataset():
    # download MNIST datasets
    data_path = "./MNIST_Data"
    if (not os.path.exists(data_path)):
        url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
              "notebook/datasets/MNIST_Data.zip"
        pth = download(url=url, path="./", kind="zip", replace=True)

    # get dataset objects
    train_data = MnistDataset(os.path.join(data_path, "train"))
    test_data = MnistDataset(os.path.join(data_path, "test"))

    # print col names
    # print(train_data.get_col_names())

    train_data = datapipe(train_data, batch_size=MnistConfig.train_batch_size)
    test_data = datapipe(test_data, batch_size=MnistConfig.test_batch_size)

    return train_data, test_data