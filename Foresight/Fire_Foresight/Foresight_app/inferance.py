from typing import Dict, Any
import lightgbm as lgb
import pandas as pd
from huggingface_hub import hf_hub_download


# Load the model once when the API starts
MODEL_PATH = hf_hub_download(
    repo_id="AchG/Fire_Foresight",
    filename="lightgbm_fire_model.txt"
)

model = lgb.Booster(model_file=MODEL_PATH)


def predict(input: Dict[str, Any]) -> Dict[str, float]:
    """
    Hugging Face Inference API will call this function.
    """
    df = pd.DataFrame([input])
    prob = float(model.predict(df)[0])

    return {"fire_risk_probability": prob}
