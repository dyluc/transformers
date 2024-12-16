import tensorflow as tf
import argparse
import os
import math

from vision_transformer import ViT
from lr_schedule import WarmupLinearDecay10

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64)
}

def normalize_image(image):
    return tf.cast(image, tf.float32) / 127.5 - 1.0 # normalize to [-1, 1]

def parse_example(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

    # decode and normalize image, and extract label
    image = tf.image.decode_jpeg(parsed_example["image"], channels=3)
    image = normalize_image(image)
    label = parsed_example["label"]
    
    return image, label
    
def main(args):
    # config
    batch_size = args.batch_size
    epochs = args.epochs
    base_learning_rate = args.base_learning_rate
    
    dataset_buffer_size = 8 * 1024 * 1024
    train_set_size = 214295
    valid_set_size = 21226
    shuffle_buffer_size = 10000
    lr_include_warmup = True
    lr_weight_decay = 1e-2
    
    steps_per_epoch = math.ceil(train_set_size / batch_size) # math#ceil for partial batches
    total_steps = steps_per_epoch * epochs
    
    # load datasets
    train_set = tf.data.TFRecordDataset(
        tf.io.gfile.glob("/opt/ml/input/data/train/*.tfrecord"), 
        compression_type="GZIP",
        buffer_size=dataset_buffer_size
    )
    
    valid_set = tf.data.TFRecordDataset(
        tf.io.gfile.glob("/opt/ml/input/data/valid/*.tfrecord"), 
        compression_type="GZIP",
        buffer_size=dataset_buffer_size
    )

    # map, shuffle, batch and prefetch
    train_set = (
        train_set
        .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    valid_set = (
        valid_set
        .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # prep callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "/opt/ml/checkpoints/vit_inat17_sagemaker/vit_inat17_epoch-{epoch:02d}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True,
    )

    tensorboard_callback=tf.keras.callbacks.TensorBoard(
        log_dir="/opt/ml/output/tensorboard", 
        histogram_freq=1
    )

    # learning rate scheduler (linear lr decay)
    if lr_include_warmup:
        lr_schedule = WarmupLinearDecay10(
            total_steps=total_steps,
            base_lr=base_learning_rate
        )
    else:
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=base_learning_rate,
            decay_steps=total_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False
        )

    # create ViT
    vit_pretraining_model = ViT(
        input_shape=(224, 224, 3),
        patch_size=16,
        num_classes=964,
        embedding_dim=256,
        num_heads=4, # multi head attention key dim 256 / 4 = 64
        num_layers=6,
        mlp_dim=1024,
        clf_mlp_dim=512,
        dropout_rate=0.1,
    )
    
    # compile
    vit_pretraining_model.compile(
        optimizer=tf.keras.optimizers.AdamW( # with weight decay (like l2 regularization)
            learning_rate=lr_schedule, # linear decay with warmup
            weight_decay=lr_weight_decay
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # train
    vit_pretraining_model.fit(
        train_set,
        validation_data=valid_set,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
        verbose=1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model directory
    parser.add_argument('--model_dir', type=str)
    
    # hyperparameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--base-learning-rate", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)