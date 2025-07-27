import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load sklearn model
model = joblib.load("model.joblib")

# --- Quantization ---
# Use min/max based scale for coef and intercept separately
coef_min = model.coef_.min()
coef_max = model.coef_.max()
inter_min = model.intercept_.min() if hasattr(model.intercept_, "__iter__") else model.intercept_
inter_max = model.intercept_.max() if hasattr(model.intercept_, "__iter__") else model.intercept_

coef_scale = (coef_max - coef_min) / 255
inter_scale = (inter_max - inter_min) / 255 if inter_max != inter_min else 1.0

# Avoid division by zero
coef_scale = coef_scale if coef_scale != 0 else 1.0

# Quantize
quant_params = {
    "coef": np.round((model.coef_ - coef_min) / coef_scale).astype(np.uint8),
    "intercept": np.round((model.intercept_ - inter_min) / inter_scale).astype(np.uint8),
    "coef_scale": coef_scale,
    "coef_min": coef_min,
    "inter_scale": inter_scale,
    "inter_min": inter_min,
}
joblib.dump(quant_params, "quant_params.joblib")

# Save original too
unquant_params = {
    "coef": model.coef_,
    "intercept": model.intercept_,
}
joblib.dump(unquant_params, "unquant_params.joblib")

# --- Dequantization ---
coef_deq = quant_params["coef"] * quant_params["coef_scale"] + quant_params["coef_min"]
intercept_deq = quant_params["intercept"] * quant_params["inter_scale"] + quant_params["inter_min"]

# --- PyTorch Model ---
class QuantModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor(coef_deq, dtype=torch.float32).unsqueeze(0))
            self.linear.bias = nn.Parameter(torch.tensor([intercept_deq], dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

# --- Inference ---
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

model_pt = QuantModel(X_test.shape[1])
model_pt.eval()

with torch.no_grad():
    preds = model_pt(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    r2 = r2_score(y_test, preds)
    print("R2 score (quantized):", r2)
