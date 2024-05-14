import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TestDataLoaderAndSplitter(unittest.TestCase):
    def setUp(self):
        #Set the data required for testing
        self.data = pd.read_csv('d:/resubmission/data/fashion-mnist_train.csv')
        self.images = self.data.iloc[:, 1:].values
        self.labels = self.data['label'].values

    def test_data_loader(self):
        # Test whether the data loader loads data correctly
        self.assertEqual(self.images.shape[0], len(self.data))
        self.assertEqual(self.images.shape[1], 28 * 28)  # 检查图像的大小

    def test_data_splitter(self):
        # Test whether the data splitter splits the data correctly
        train_images, validation_images, train_labels, validation_labels = train_test_split(
            self.images, self.labels, test_size=0.2, random_state=42)
        self.assertEqual(train_images.shape[0], len(train_labels))
        self.assertEqual(validation_images.shape[0], len(validation_labels))
        
    def test_model_structure(self):
        model = Sequential([
            Input(shape=(28, 28, 1)),
            MyConv2D(filters=32, kernel_size=5),
            MaxPooling2D(pool_size=(2, 2)),
            MyConv2D(filters=64, kernel_size=5),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=1024, activation='relu'),
            Dropout(0.5),
            Dense(units=10, activation='softmax')
        ])
        self.assertEqual(len(model.layers), 8)

    def test_model_compilation(self):
        model = Sequential([
            Input(shape=(28, 28, 1)),
            MyConv2D(filters=32, kernel_size=5),
            MaxPooling2D(pool_size=(2, 2)),
            MyConv2D(filters=64, kernel_size=5),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=1024, activation='relu'),
            Dropout(0.5),
            Dense(units=10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam')

    def test_model_training(self):
        images = np.random.rand(1000, 28, 28, 1)
        labels = to_categorical(np.random.randint(0, 10, size=1000), num_classes=10)
        
        model = Sequential([
            Input(shape=(28, 28, 1)),
            MyConv2D(filters=32, kernel_size=5),
            MaxPooling2D(pool_size=(2, 2)),
            MyConv2D(filters=64, kernel_size=5),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=1024, activation='relu'),
            Dropout(0.5),
            Dense(units=10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(images, labels, epochs=2, batch_size=32, validation_split=0.2, verbose=0)
        self.assertEqual(len(history.history['loss']), 2)

if __name__ == '__main__':
    unittest.main()
