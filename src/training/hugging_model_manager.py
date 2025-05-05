import os
import torch
from pathlib import Path
from datetime import datetime, date
from huggingface_hub import HfApi, HfFolder
from transformers import SegformerForSemanticSegmentation

from ..ConfigParser import ConfigParser
from ..utils.training_step import TrainingStep, resume_from_training_step

class ModelManager():

    def __init__(self, cp: ConfigParser, ts: TrainingStep):
        self.cp = cp
        self.training_step= ts

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.resume_from_checkpoint, self.latest_checkpoint = None, None

        self.model, self.output_dir, self.model_name_with_username = None, None, None


    def push_to_hub(self) -> bool:
        return False if self.training_step == TrainingStep.COARSE else self.cp.upload_on_huggingface

    def setup_model_dir(self) -> None:
        """ Return session_name and output_dir computed by args. """

        # Build a unique model name.
        now = datetime.now()
        elapsed_second_in_day = now.hour * 3600 + now.minute * 60 + now.second
        today = f'{date.today().strftime("%Y_%m_%d")}_{elapsed_second_in_day}'
        training_type = "_coarse" if self.training_step == TrainingStep.COARSE else "_refine"

        self.model_name = f"{self.cp.model_name}-{today}-bs{self.cp.batch_size}{training_type}"[0:96] # Add a limit for huggingface
        self.output_dir = Path(self.cp.path_models_checkpoints, self.model_name)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check for resume.
        resume_str = resume_from_training_step(self.cp, self.training_step)

        if resume_str == None: return
        resume_path = Path(resume_str)
        if not resume_path.exists():
            print(f"Resume path not found at {resume_path}, retrain from scratch")
            return
        
        self.model_name = resume_path.name
        self.output_dir = resume_path
        checkpoints = [f for f in self.output_dir.iterdir() if f.name.startswith('checkpoint')]
        if len(checkpoints) != 0:
            self.latest_checkpoint = max(checkpoints, key=os.path.getctime)
            self.resume_from_checkpoint = self.latest_checkpoint
            

    def setup_model(self, id2labels: dict) -> None:

        label2id = {v: k for k, v in id2labels.items()}
    
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.cp.base_model_name,
            id2label=id2labels,
            label2id=label2id,
            num_labels=len(id2labels),  # Single channel output for fuzzy mask prediction
            ignore_mismatched_sizes=True  # Allow resizing output layers
        ).to(self.device)
    

    def send_data_to_hugging_face(self) -> None:
        """ Send files to hugging face."""
        
        # Send data to huggingface.
        token = HfFolder.get_token()
        hf_api = HfApi(token=token)
        try:
            username = hf_api.whoami()["name"]
        except:
            print("User not found with hugging face token provide.")
            return
        
        repo_id = f"{username}/{self.model_name}"
        try:
            repo_url = hf_api.create_repo(token=token, repo_id=repo_id, private=False, exist_ok=True)
            print(f"Repository URL: {repo_url}")
        except Exception as e:
            raise NameError(f"Error creating repository: {e}")

        all_files = [f for f in self.output_dir.iterdir() if f.is_file() and f.name != "model.safetensors"]

        for filepath in all_files:
            hf_api.upload_file(
                token=token,
                path_or_fileobj=filepath,
                path_in_repo=filepath.name,
                repo_id=self.model_name_with_username,
                commit_message=f"Upload {filepath.name}"
            )

        print(f"All files successfully uploaded to the Hub: {repo_url}")


    def get_hf_username(self) -> str:
        hf_api = HfApi(token=self.cp.hugging_face_token)
        try:
            username = hf_api.whoami()["name"]
        except:
            raise NameError("User not found with hugging face token provide.")

        return username
    
    def get_model_name_with_username(self) -> str:
        if self.model_name_with_username == None:
            self.model_name_with_username = f"{self.get_hf_username()}/{self.model_name}"
        
        return self.model_name_with_username