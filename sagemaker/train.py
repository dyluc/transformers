import tensorflow as tf
import argparse
import os

from vision_transformer import ViT

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64)
}

def normalize_image(image):
    return tf.cast(image, tf.float32) / 127.5 - 1.0 # normalize to [-1, 1]

def parse_example_safely(serialized_example):
    try:
        parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    
        # decode and normalize image, and extract label
        image = tf.image.decode_jpeg(parsed_example["image"], channels=3)
        image = normalize_image(image)
        label = parsed_example["label"]
        
        return image, label
    except tf.errors.InvalidArgumentError as e:
        tf.print(f"Error parsing example: {e}")
        return None, None
            
def main(args):
    # load datasets
    dataset_buffer_size = 8 * 1024 * 1024
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
    batch_size = args.batch_size
    buffer_size = 10000
    train_set = (
        train_set
        .map(parse_example_safely, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda image, label: image is not None and label is not None)
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    valid_set = (
        valid_set
        .map(parse_example_safely, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda image, label: image is not None and label is not None)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # 20% train: 115836 val: 19197
    train_set = train_set.take(115836)
    valid_set = valid_set.take(19197)
    #

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
        patience=5,
        restore_best_weights=True,
    )
    
    train_set_size = 115836
    epochs = args.epochs
    linear_lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.base_learning_rate,
        decay_steps=(train_set_size // batch_size) * epochs, # base LR * (1 - t / T)
        end_learning_rate=0.0,
    )

    # create ViT
    vit_pretraining_model = ViT(
        input_shape=(224, 224, 3),
        patch_size=16,
        num_classes=5089,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        mlp_dim=1024,
        dropout_rate=0.1,
    )
    
    # compile
    vit_pretraining_model.compile(
        optimizer=tf.keras.optimizers.AdamW( # with weight decay (like l2 regularization)
            learning_rate=linear_lr_decay, # we can reduce base lr or weight decay if unstable during training
            weight_decay=0.1
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # train
    vit_pretraining_model.fit(
        train_set,
        validation_data=valid_set,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        verbose=1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--base-learning-rate", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)