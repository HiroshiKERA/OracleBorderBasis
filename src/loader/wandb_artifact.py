import wandb
from transformers import AutoModelForSequenceClassification
import os 



def load_wandb_model(model_name: str, model_version: str, project: str = "transformer-border", group: str = "dump"):
    # Create a new run
    with wandb.init(project=project, group=group) as run:
        # Specify artifact name and version
        model_name = f"{model_name}:{model_version}"
        model_artifact = run.use_artifact(model_name)

        # Download model weights to folder and return path
        model_dir = model_artifact.download()

        # Load Hugging Face model from folder using same model class
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        )

    return model

if __name__ == "__main__":
    model_name = "model-expansion_base_k_lt3_m1000000_GF31_n3_deg1_terms10_bounds4_4_4_total4_20250429_152414"
    model_version = "latest"
    load_wandb_model(model_name, model_version, 
                     project="transformer-border", group="dump")