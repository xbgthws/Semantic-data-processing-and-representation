import time, os
import tensorflow as tf
from transformer import Transformer
from data_processing import load_data, DataGenerator, tokenize
from utils import create_masks
from tqdm import tqdm
import config

# load data
inp_text, tar_text = load_data(file_path="./data/xiaohuangji.tsv", num_samples=config.num_samples)
inp_seq, tar_seq, _, _, _ = tokenize(inp_text, tar_text)

train_dataset = DataGenerator(tokenizer_data=(inp_seq, tar_seq), batch_size=config.batch_size)

# define model
transformer = Transformer(**config.model_params)

if os.path.exists(os.path.join(config.BASE_MODEL_DIR, 'transformer_weights.h5')):
    transformer.load_weights(os.path.join(config.BASE_MODEL_DIR, 'transformer_weights.h5'))

# define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(**config.optimizer_params)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]  # input of decoder
    tar_real = tar[:, 1:]  # output of decoder

    # creating  mask
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # calculating the gradient
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True,
                                     enc_padding_mask,
                                     combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


EPOCHS = config.n_epoch

for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    progress = tqdm(
        train_dataset,
        total=len(train_dataset),
        desc=f'Epoch {epoch + 1}/{EPOCHS}',
        unit_scale=True
    )

    for (batch, (inp, tar)) in enumerate(progress):
        train_step(inp, tar)

        if batch % 100 == 0:
            progress.set_postfix({'Loss': train_loss.result().numpy(),
                                  'Accuracy': train_accuracy.result().numpy()})

    progress.close()

    progress.write('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, train_loss.result(), train_accuracy.result()))
    progress.write(f'Time taken for 1 epoch: {time.time() - start} secs\n')

    if not os.path.exists(config.BASE_MODEL_DIR):
        os.makedirs(config.BASE_MODEL_DIR)
    transformer.save_weights(os.path.join(config.BASE_MODEL_DIR, 'transformer.h5'))  # save model
