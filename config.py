import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 2
IMAGE_SHAPE = (1, 28, 28)
CONVOLVED_SHAPE = (128, 4, 4)
TRAIN_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
MSE_LOSS_SCALE = 500
DATA_PATH = "./data/"
MODELS_PATH = "./models/"
OUTPUT_PATH = "./output/"
