from setuptools import setup, find_packages

setup(
    name="cifar-playground",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "draw-images=model.visualize:draw_images",
            "draw-cnn-features=model.visualize:draw_cnn_features",
            "train-shallow=model.train:train_shallow_model",
            "train=model.train:train_model",
        ],
    }
)
