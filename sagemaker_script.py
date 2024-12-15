import sagemaker
from sagemaker.tensorflow import TensorFlow

train_data = "s3://inat17-train-val-records/train_val_images-processed/train2017/"
valid_data = "s3://inat17-train-val-records/train_val_images-processed/val2017/"
output_path = "s3://inat17-vit-model-artifacts/"

tf_estimator = TensorFlow(
    entry_point="train.py",
    source_dir="./sagemaker",
    role="arn:aws:iam::470419151546:role/SageMakerRoleViTTraining",
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    framework_version="2.16",
    py_version="py310",
    hyperparameters={
        "batch-size": 16,
        "epochs": 8,
        "base-learning-rate": 1e-4,
    },
    output_path=output_path,
    checkpoint_s3_uri=output_path,
    volume_size=30,
    script_mode=True
)

# Launch training
tf_estimator.fit({
    "train": train_data,
    "valid": valid_data
})
