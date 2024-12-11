# Preparing iNaturalist2017 Training and Validation Set

Notes:
- `prep_sets.py` just runs some preprocessing on the JSON files for easier access to image filenames and labels.
- There is class imbalance in the dataset, distribution of images per category follows the observation frequency of that category. We can consider techniques (oversampling, smote, etc) to manage classes with few examples following analysis of the confusion matrix.
- On the resizing strategy, padding minimizes subject loss and maintains aspect ratio. Distortion via resizing or potentially cropping out important features is more risky for ViTs than padding. Attention mechanisms can learn to ignore padding regions (even without configuring any attention masks). We can inspect our attention maps to ensure these areas receive low attention scores. Hopefully the ViT can use the positional embeddings (and an understanding of the spatial arrangement of the patches) to differentiate padded regions from patches with actual content. All non square images will have padded regions.

## Data Preprocessing
- Load and process JSON files into just filename and category ids, then shuffle on disk.
- Load processed JSON files and batch the annotations, attaching the labels. For each annotation batch, construct a dataset, load the images, resize and normalize them, and finally shuffle the batch.
- Serialize batch of images to single TFRecord file, repeat for all batches in the given set.
- Shuffle on disk.

Then the processed datasets can be uploaded to S3 for training in SaturnCloud. Data augmentation can be applied above too, different strategies can be used for cropping (even using bounding boxes where present), though padding seems most sensible and straightforward given the model architecture.

## Data Ingestion

- Enabling parallel reading and prefetching. Tensorflow datasets can be created from Python generators. A dedicated generator can manage downloading groups of our preprocessed TFRecord batch files, and deleting them once used.
- The pipeline will include interleaving, further shuffling, batching and prefetching configurations.
- This is a flexible approach for environments with constrained disk space, such as a SaturnCloud server.

> [!NOTE]
> If you have sufficient disk space to store your entire training dataset locally, it is generally more
> efficient to do so, as it avoids potential bottlenecks that could be introduced downloading data during
> training. The above is just a flexible approach for environments with limited disk space or when
> working with large datasets that cannot be stored entirely on disk. See `ingestion-pipeline.ipynb` for an
> implementation. S3 incurs data transfer costs so be careful how much data you transfer. 

## Dataset Corrupt JPEGs

I found during dataset preperation that there are a few corrupt files across the validation and training sets. Since there were so few, I located the relevant observations on https://www.inaturalist.org/ and replaced them in the set (using the category and rights holder information attached to the annotations).

Validation (Under annotation batches 66 and 87):
- `train_val_images/Plantae/Yucca schidigera/e86b5b9e546b87b003f433ae6c09e15d.jpg`
- `train_val_images/Plantae/Tetraneuris scaposa/5d7d2fc611977e600ceb14de7444667a.jpg`

Training (Under annotation batch 22):

- `train_val_images/Plantae/Eschscholzia californica/006fa2e4d3c83014333f7203a84fff8c.jpg`