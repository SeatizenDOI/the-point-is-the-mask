import json
import torch
import datasets
import tokenizers
import transformers
from pathlib import Path


from .hugging_model_manager import ModelManager
from ..ConfigParser import ConfigParser


def generate_model_card(data_paths: list[Path], model_manager: ModelManager) -> None:

    data = {}
    for data_path in data_paths:
        if not data_path.exists(): 
            data[data_path.stem] = {}
            continue
        with open(data_path, 'r') as file:
            data[data_path.stem] = json.load(file)
  
    markdown_training_results = format_training_results_to_markdown(data["trainer_state"])
    hyperparameters_markdown = format_hyperparameters_to_markdown(data["config"], data["all_results"])
    framework_versions_markdown = format_framework_versions_to_markdown()  

    markdown_content = f"""
---
language:
- eng
license: cc0-1.0
tags:
- segmentic-segmentation
- generated_from_trainer
base_model: {model_manager.model_name}
model-index:
- name: {model_manager.output_dir.name}
  results: []
---

{model_manager.model_name} is a fine-tuned version of [{model_manager.cp.base_model_name}](https://huggingface.co/{model_manager.cp.base_model_name}).

---

# Model description
{model_manager.model_name} is a model built on top of {model_manager.cp.base_model_name} model for underwater multilabel image classification.The classification head is a combination of linear, ReLU, batch normalization, and dropout layers.
\nThe source code for training the model can be found in this [Git repository](https://github.com/SeatizenDOI/DinoVdeau).

- **Developed by:** [lombardata](https://huggingface.co/lombardata), credits to [César Leblanc](https://huggingface.co/CesarLeblanc) and [Victor Illien](https://huggingface.co/groderg)

---

# Intended uses & limitations
You can use the raw model for classify diverse marine species, encompassing coral morphotypes classes taken from the Global Coral Reef Monitoring Network (GCRMN), habitats classes and seagrass species.



---

# Training procedure

## Training hyperparameters
{hyperparameters_markdown}


## Training results
{markdown_training_results}

---

# Framework Versions
{framework_versions_markdown}
"""

    output_filename = "README.md"
    with open(Path(model_manager.output_dir, output_filename), 'w') as file:
        file.write(markdown_content)

    print(f"Model card generated and saved to {output_filename} in the directory {model_manager.output_dir}")


def format_transforms_to_markdown(transforms_dict):
    transforms_markdown = "\n"
    for key, value in transforms_dict.items():
        transforms_markdown += f"{key.replace('_', ' ').title()}\n"
        for item in value:
            probability = item.get('probability', 'No additional parameters')
            if isinstance(probability, float):
                probability = f"probability={probability:.2f}"
            transforms_markdown += f"- **{item['operation']}**: {probability}\n"
        transforms_markdown += "\n"
    return transforms_markdown


def format_training_results_to_markdown(trainer_state: dict) -> str:
    training_logs = trainer_state.get("log_history", [])

    markdown_table = "Epoch | Validation Loss | Learning Rate\n"

    markdown_table += "--- | --- | ---\n"

    
    seen_epochs = set()

    for log in training_logs:
        if "eval_loss" not in log: continue

        epoch = log.get("epoch", "N/A")
        epoch = int(epoch)  # Ensure epoch is displayed as an integer
        if epoch in seen_epochs:
            continue  # Skip this log if the epoch has already been added
        seen_epochs.add(epoch)

        validation_loss = log.get("eval_loss", "N/A")
        learning_rate = log.get("learning_rate", "N/A")
        markdown_table += f"{epoch} | {validation_loss} | {learning_rate}\n"
    
    return markdown_table


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
    markdown += f"- **Freeze Encoder**: {'Yes' if config.get('freeze_encoder', False) else 'No'}\n"
    markdown += f"- **Data Augmentation**: {'Yes' if config.get('data_augmentation', False) else 'No'}\n"
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


def save_hyperparameters_to_config(output_dir: Path, cp: ConfigParser) -> None:

    # Regroup and save hyperparameters
    hyperparameters = {
        'initial_learning_rate': cp.initial_learning_rate,
        'train_batch_size': cp.batch_size,
        'eval_batch_size': cp.batch_size,
        'optimizer': {'type': 'Adam'},
        'lr_scheduler_type': {'type': 'ReduceLROnPlateau'},
        'patience_lr_scheduler': cp.patience_lr_scheduler,
        'factor_lr_scheduler': cp.factor_lr_scheduler,
        'weight_decay': cp.weight_decay,
        'early_stopping_patience': cp.early_stopping_patience,
        'num_epochs': cp.epochs
    }
    
    # Load hyperparameters.
    config_path, config = Path(output_dir, 'config.json'), {}
    if Path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)

    # Save hyperparameters.
    config.update(hyperparameters)
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    print("Updated configuration saved to config.json")