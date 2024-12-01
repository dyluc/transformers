import json
import tensorflow as tf

def process_and_save_json(dataset_type):
    with open(f"./train_val2017/{dataset_type}.json", "r") as f:
        dataset = json.load(f)
    
    images = dataset["images"]
    annotations = dataset["annotations"]
    categories = dataset["categories"]
    
    assert len(images) == len(annotations)
    print(f"Examples in {dataset_type}: {len(images)}.")
    
    # category dict for lookup
    cat_dict = {}
    for category in categories:
        cat_dict[category["id"]] = f"{category["supercategory"]}/{category["name"]}"
        
    example_annotations = []
    for i in range(len(images)):
        file_name = images[i]["file_name"]
        category_id = annotations[i]["category_id"]    
        example = {
            "file_name": file_name,
            "category_id": category_id,
        }
        example_annotations.append(example)
    
    # quick final validation
    assert len(example_annotations) == len(images)
    for i, example in enumerate(example_annotations):
        assert example["file_name"] == images[i]["file_name"]
        assert example["category_id"] == annotations[i]["category_id"]
        assert images[i]["id"] == annotations[i]["image_id"]
    
    print("All filenames, category & image ids match.")
        
    with open(f"./train_val2017/{dataset_type}-processed.json", "w") as f:
        json.dump(example_annotations, f)
    
    print("File write finished.")

# process dataset annotations into more convenient files
dataset_type = "train2017"
process_and_save_json(dataset_type)