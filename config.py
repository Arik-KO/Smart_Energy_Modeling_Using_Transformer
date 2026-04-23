import torch

WINDOW_SIZE = 24
DROPOUT = 0.1
ENCODER_LAYERS = 2
EMBEDDING_DIM = 16
D_FF = 128
BATCH_SIZE = 64
EPOCHS = 20
HEADS = 4
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')