import torch
import torch.nn as nn

from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer, EarlyStoppingCallback

from .loss import DiceLoss
from ..ConfigParser import ConfigParser

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_function: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()  # Convert labels to integers
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [B, num_labels, H, W]

        # Resize labels to match prediction dimensions
        labels = nn.functional.interpolate(
            labels.unsqueeze(1).float(),  # Convert to (B, 1, H, W)
            size=logits.shape[-2:],  # Match logits spatial size
            mode="nearest"
        ).squeeze(1).long()  # Convert back to (B, H, W) as integers

        loss = self.loss_function(logits, labels)

        return (loss, outputs) if return_outputs else loss



def setup_trainer(cp: ConfigParser, num_labels: int, train_ds, validation_ds, ) -> CustomTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        cp.model_name,
        num_labels=num_labels,  # Single channel output for fuzzy mask prediction
        ignore_mismatched_sizes=True  # Allow resizing output layers
    ).to(device)

    training_args = TrainingArguments(
        output_dir=cp.path_output_dir,
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
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
        remove_unused_columns=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
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
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        callbacks=[early_stop],
        optimizers=(optimizer, lr_scheduler),
        loss_function=DiceLoss()
    )

    return trainer