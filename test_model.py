import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TestDataLoaderAndSplitter(unittest.TestCase):
    def setUp(self):
        # 设置测试所需的数据
        self.data = pd.read_csv('d:/resubmission/data/fashion-mnist_train.csv')
        self.images = self.data.iloc[:, 1:].values
        self.labels = self.data['label'].values

    def test_data_loader(self):
        # 测试数据加载器是否正确加载数据
        self.assertEqual(self.images.shape[0], len(self.data))
        self.assertEqual(self.images.shape[1], 28 * 28)  # 检查图像的大小

    def test_data_splitter(self):
        # 测试数据分割器是否正确分割数据
        train_images, validation_images, train_labels, validation_labels = train_test_split(
            self.images, self.labels, test_size=0.2, random_state=42)
        self.assertEqual(train_images.shape[0], len(train_labels))
        self.assertEqual(validation_images.shape[0], len(validation_labels))

if __name__ == '__main__':
    unittest.main()