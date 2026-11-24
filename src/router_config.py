
import yaml

from src.common import ROOT

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
ARTIFACTS = ROOT / "artifacts/support_router"
MODELS = ARTIFACTS / "models"
METRICS = ARTIFACTS / "metrics"
LOGS = ARTIFACTS / "logs"
BASELINE = MODELS / "baseline.joblib"
DISTILBERT = MODELS / "distilbert"
INITIAL = MODELS / "distilbert_initial"
TRAIN_LOCK = LOGS / "training.lock"
