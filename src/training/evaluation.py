from datetime import date
from datasets import Dataset

from .trainer import CustomTrainer

def evaluate_and_save(trainer: CustomTrainer, ds_test: Dataset, push_to_hub: bool) -> None:
    metrics = trainer.evaluate(ds_test)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    if not push_to_hub: return

    today = date.today().strftime("%Y_%m_%d")
    try:
        trainer.push_to_hub(commit_message=f"Evaluation on the test set completed on {today}.")
    except Exception as e:
            print(f"Error while pushing to Hub: {e}")

