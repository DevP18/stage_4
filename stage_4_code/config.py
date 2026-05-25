import torch

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 10

SEQ_LENGTH = 200

EMBEDDING_DIM = 128

HIDDEN_DIM = 128

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)