all: 
	@echo "Drawing first 10 samples"
	-draw-images --datapath data/cifar-10-batches-py/

	@echo "Drawing CNN features 2D embedding"
	-draw-cnn-features --datapath $<

	@echo "Training shallow model"
	-train-shallow --datapath $<

	@echo "Training full model"
	-train --datapath $<

.PHONY: all
