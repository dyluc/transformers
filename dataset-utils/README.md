# Preparing iNaturalist2017 Training and Validation Set

Notes:
- `prep_sets.py` just runs some preprocessing on the JSON files for easier access to image filenames and labels.
- There is class imbalance in the dataset, distribution of images per category follows the observation frequency of that category. Consider techniques (oversampling, smote, etc) to manage classes with few examples following analysis of confusion matrix.
- On the resizing strategy, padding minimizes subject loss and maintains aspect ratio. Distortion via resizing or potentially cropping out important features is more risky for ViTs than padding. Attention mechanisms can learn to ignore padding regions (even without configuring any attention masks). We can inspect our attention maps to ensure these areas receive low attention scores. Hopefully the ViT can use the positional embeddings (and an understanding of the spatial arrangement of the patches) to differentiate padded regions from patches with actual content. All non square images will have padded regions.

Plan for data preprocessing (repeat the following for both training and validation sets):
- Load and process JSON files into just filename and category ids, then shuffle on disk.
- Load processed JSON files and batch the annotations, attaching the labels. For each annotation batch, construct a dataset, load the images, resize and normalize them, and finally shuffle the batch.
- Serialize batch of images to single TFRecord file, repeat for all batches in the given set.
- Shuffle on disk.

Then the processed datasets can be uploaded to S3 for training in SaturnCloud. Data augmentation can be applied above too, different strategies can be used for cropping (even using bounding boxes where present), though padding seems most sensible and straightforward given the model architecture.

Plan for data ingestion pipeline:

- Create data ingestion pipeline, to read in and train from TFRecord files (for specific load size + parallel reading and prefetching), pause training, reload next chunk from S3, repeat (don't want to load all data onto disk in SaturnCloud). In pipeline include interleaving (see book), shuffling, batching and prefetching configurations.

## Dataset Corrupt JPEGs

I found during dataset preperation that there are a few corrupt files across the validation and training sets. Since there were so few, I located the relevant observations on https://www.inaturalist.org/ and replaced them in the set (using the category and rights holder information attached to the annotations).

Validation (Under annotation batches 66 and 87):
- `train_val_images/Plantae/Yucca schidigera/e86b5b9e546b87b003f433ae6c09e15d.jpg`
- `train_val_images/Plantae/Tetraneuris scaposa/5d7d2fc611977e600ceb14de7444667a.jpg`

Training (Under annotation batch 22):

- `train_val_images/Plantae/Eschscholzia californica/006fa2e4d3c83014333f7203a84fff8c.jpg`