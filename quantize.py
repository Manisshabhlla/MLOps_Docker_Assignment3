# quantize.py
import joblib
import numpy as np
import torch
import torch.nn as nn

# Load sklearn model
model = joblib.load("model.joblib")

# Save original weights
unquant_params = {
    "coef": model.coef_,
    "intercept": model.intercept_
}
joblib.dump(unquant_params, "unquant_params.joblib")

# Quantize
scale = 255 / (np.max(model.coef_) - np.min(model.coef_))
quant_params = {
    "coef": np.round((model.coef_ - np.min(model.coef_)) * scale).astype(np.uint8),
    "intercept": np.round((model.intercept_ - np.min(model.coef_)) * scale).astype(np.uint8),
    "scale": scale,
    "min": np.min(model.coef_)
}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantize
coef_deq = quant_params["coef"] / scale + quant_params["min"]
intercept_deq = quant_params["intercept"] / scale + quant_params["min"]

# PyTorch model
class QuantModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor(coef_deq).unsqueeze(0))
            self.linear.bias = nn.Parameter(torch.tensor([intercept_deq]))

    def forward(self, x):
        return self.linear(x)

# Inference
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model_pt = QuantModel(X_test.shape[1])
model_pt.eval()

with torch.no_grad():
    preds = model_pt(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    print("R2 score (quantized):", r2_score(y_test, preds))
