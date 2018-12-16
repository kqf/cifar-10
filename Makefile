data_url = https://www.cs.toronto.edu/~kriz/
data_file = cifar-10-python.tar.gz

.PHONY: all

all: data/cifar-10-batches-py
	@echo "Drawing first 10 samples"
	draw-images --datapath data/cifar-10-batches-py/

	@echo "Drawing CNN features 2D embedding"
	draw-cnn-features --datapath $<

	@echo "Training shallow model"
	train-shallow --datapath $<

	@echo "Training full model"
	train --datapath $<

data/cifar-10-batches-py:
	echo "Downloading data"
	wget $(data_url)/$(data_file) -O data/$(data_file)
	unar -D data/$(data_file) -o data/
