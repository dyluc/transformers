import os
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import TensorBoardOutputConfig

train_data = "s3://inat17-train-val-records/train_val_images-processed(Aves)/train2017"
valid_data = "s3://inat17-train-val-records/train_val_images-processed(Aves)/val2017"
output_path = "s3://inat17-vit-model-artifacts"

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=f"{output_path}/tensorboard",
    container_local_output_path="/opt/ml/output/tensorboard"
)

tf_estimator = TensorFlow(
    entry_point="train.py",
    source_dir="./sagemaker",
    role="arn:aws:iam::470419151546:role/SageMakerRoleViTTraining",
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    framework_version="2.16",
    py_version="py310",
    hyperparameters={
        "batch-size": 32,
        "epochs": 10,
        "base-learning-rate": 5e-4,
    },
    output_path=output_path,
    checkpoint_s3_uri=f"{output_path}/checkpoints",
    tensorboard_output_config=tensorboard_output_config,
    volume_size=30,
    script_mode=True
)

# Launch training
tf_estimator.fit({
    "train": train_data,
    "valid": valid_data
})
