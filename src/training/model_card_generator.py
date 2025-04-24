from pathlib import Path
import json

import transformers
import tokenizers
import datasets
import torch

def generate_model_card(data_paths: list[Path], modelManager: HuggingModelManager, datasetManager: DatasetManager) -> None:

    data = {}
    for data_path in data_paths:
        if not data_path.exists(): 
            data[data_path.stem] = {}
            continue
        with open(data_path, 'r') as file:
            data[data_path.stem] = json.load(file)
  
    markdown_training_results = format_training_results_to_markdown(data["trainer_state"], datasetManager.label_type)
    test_results_metrics_markdown = extract_test_results(data["test_results"], datasetManager.label_type, data["test_f1_per_class"])
    markdown_counts = datasetManager.counts_df.to_markdown(index=False)
    transforms_markdown = format_transforms_to_markdown(data["transforms"])
    if data["config"].get('data_augmentation') == False :
        transforms_markdown = "No augmentation"
    hyperparameters_markdown = format_hyperparameters_to_markdown(data["config"], data["all_results"])
    framework_versions_markdown = format_framework_versions_to_markdown()  

    markdown_content = f"""
---
language:
- eng
license: cc0-1.0
tags:
- {modelManager.args.training_type}-image-classification
- {modelManager.args.training_type}
- generated_from_trainer
base_model: {modelManager.model_name}
model-index:
- name: {modelManager.output_dir.name}
  results: []
---

{modelManager.args.new_model_name} is a fine-tuned version of [{modelManager.model_name}](https://huggingface.co/{modelManager.model_name}). It achieves the following results on the test set:

{test_results_metrics_markdown}

---

# Model description
{modelManager.args.new_model_name} is a model built on top of {modelManager.model_name} model for underwater multilabel image classification.The classification head is a combination of linear, ReLU, batch normalization, and dropout layers.
\nThe source code for training the model can be found in this [Git repository](https://github.com/SeatizenDOI/DinoVdeau).

- **Developed by:** [lombardata](https://huggingface.co/lombardata), credits to [César Leblanc](https://huggingface.co/CesarLeblanc) and [Victor Illien](https://huggingface.co/groderg)

---

# Intended uses & limitations
You can use the raw model for classify diverse marine species, encompassing coral morphotypes classes taken from the Global Coral Reef Monitoring Network (GCRMN), habitats classes and seagrass species.

---

# Training and evaluation data
Details on the {'' if datasetManager.label_type == LabelType.BIN else 'estimated'} number of images for each class are given in the following table:
{markdown_counts}

---

# Training procedure

## Training hyperparameters
{hyperparameters_markdown}

## Data Augmentation
Data were augmented using the following transformations :
{transforms_markdown}

## Training results
{markdown_training_results}

---

# Framework Versions
{framework_versions_markdown}
"""

    output_filename = "README.md"
    with open(Path(modelManager.output_dir, output_filename), 'w') as file:
        file.write(markdown_content)

    print(f"Model card generated and saved to {output_filename} in the directory {modelManager.output_dir}")



def format_hyperparameters_to_markdown(config, all_results):
    epoch = all_results.get("epoch", None)
    if epoch == None:
        epoch = config.get('num_epochs', 'Not specified')

    markdown = "\n"
    markdown += "The following hyperparameters were used during training:\n\n"
    markdown += f"- **Number of Epochs**: {epoch}\n"
    markdown += f"- **Learning Rate**: {config.get('initial_learning_rate', 'Not specified')}\n"
    markdown += f"- **Train Batch Size**: {config.get('train_batch_size', 'Not specified')}\n"
    markdown += f"- **Eval Batch Size**: {config.get('eval_batch_size', 'Not specified')}\n"
    markdown += f"- **Optimizer**: {config.get('optimizer', {}).get('type', 'Not specified')}\n"
    markdown += f"- **LR Scheduler Type**: {config.get('lr_scheduler_type', {}).get('type', 'Not specified')} with a patience of {config.get('patience_lr_scheduler', 'Not specified')} epochs and a factor of {config.get('factor_lr_scheduler', 'Not specified')}\n"
    markdown += f"- **Freeze Encoder**: {'Yes' if config.get('freeze_encoder', True) else 'No'}\n"
    markdown += f"- **Data Augmentation**: {'Yes' if config.get('data_augmentation', True) else 'No'}\n"
    return markdown


def format_framework_versions_to_markdown():
    transformers_version = transformers.__version__
    pytorch_version = torch.__version__
    datasets_version = datasets.__version__
    tokenizers_version = tokenizers.__version__

    markdown = "\n"
    markdown += f"- **Transformers**: {transformers_version}\n"
    markdown += f"- **Pytorch**: {pytorch_version}\n"
    markdown += f"- **Datasets**: {datasets_version}\n"
    markdown += f"- **Tokenizers**: {tokenizers_version}\n"
    return markdown
