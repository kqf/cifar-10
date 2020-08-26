all: 
	@echo "Drawing first 10 samples"
	-draw-images

	@echo "Drawing CNN features 2D embedding"
	-draw-cnn-features

	@echo "Training shallow model"
	-train-shallow

	@echo "Training full model"
	-train-cnn-features

	@echo "Training the deep model"
	-train-cnn-features

.PHONY: all
