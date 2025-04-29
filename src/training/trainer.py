import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback


from .loss import DiceLoss
from .dataset import DatasetManager
from .hugging_model_manager import ModelManager
from ..ConfigParser import ConfigParser


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_function: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()  # Convert labels to integers
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [B, num_labels, H, W]

        # Resize labels to match prediction dimensions
        labels = torch.nn.functional.interpolate(
            labels.unsqueeze(1).float(),  # Convert to (B, 1, H, W)
            size=logits.shape[-2:],  # Match logits spatial size
            mode="nearest"
        ).squeeze(1).long()  # Convert back to (B, H, W) as integers

        loss = self.loss_function(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # Ensure evaluations are logged only at the end of each epoch
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs, start_time)



def setup_trainer(cp: ConfigParser, dataset_manager: DatasetManager, model_manager: ModelManager) -> CustomTrainer:

    training_args = TrainingArguments(
        output_dir=model_manager.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",  # Log only once per epoch
        learning_rate=cp.initial_learning_rate,
        per_device_train_batch_size=cp.batch_size,
        per_device_eval_batch_size=cp.batch_size,
        num_train_epochs=cp.epochs,
        weight_decay=cp.weight_decay,
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_dir = Path(model_manager.output_dir, "logs"),
        logging_steps=10,
        report_to="tensorboard",
        push_to_hub=model_manager.push_to_hub(),
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
        remove_unused_columns=False
    )

    optimizer = torch.optim.Adam(
        model_manager.model.parameters(),
        lr=cp.initial_learning_rate,
        weight_decay=cp.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cp.factor_lr_scheduler,
        patience=cp.patience_lr_scheduler
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=cp.early_stopping_patience)

    trainer = CustomTrainer(
        model=model_manager.model,
        args=training_args,
        train_dataset=dataset_manager.train_ds,
        eval_dataset=dataset_manager.validation_ds,
        callbacks=[early_stop],
        optimizers=(optimizer, lr_scheduler),
        loss_function=DiceLoss()
    )

    return trainer