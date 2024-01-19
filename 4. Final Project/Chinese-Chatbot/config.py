from utils import CustomizedSchedule

BASE_MODEL_DIR = "./saved_models"

# Dataset parameters
num_samples = None  # Number of samples read, None means reading all samples
vocab_path = "./data/vocab.pkl"
vocab_size = 6346

n_epoch = 5
batch_size = 128
d_model = 512  # Dimensions of word embeddings


model_params = {
    "num_layers": 4,  # number of layers in encoder and decoder
    "d_model": d_model,  # Dimension of the word embedding
    "num_heads": 8,  # Number of heads for multi-head attention
    "dff": 512,  # Dimension of the middle layer of the feedforward neural network
    "input_vocab_size": vocab_size,  # Input vocabulary size
    "target_vocab_size": vocab_size,  # Output vocabulary size
    "pe_input": 241,  # Maximum length of the input sequence
    "pe_target": 250,  # Maximum length of the output sequence.
    "rate": 0.1  # dropout retention rate
}


optimizer_params = {
    "learning_rate": CustomizedSchedule(d_model),
    "beta_1": 0.9,
    "beta_2": 0.98,
    "epsilon": 1e-9
}
