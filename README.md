# MLOps Assignment 3

This project shows how to build a full machine learning pipeline that runs automatically.

We:
- Train a machine learning model
- Put it inside a Docker container
- Set up GitHub Actions to train and test it automatically
- Then convert the model to a smaller size using quantization

### 1. Model Training (`dev` branch)
- Used the **California Housing dataset**
- Trained a **Linear Regression model**
- Saved the model as `model.joblib`

### 2. Docker + CI/CD (`docker_ci` branch)
- Created a **Dockerfile**
- Wrote a script `predict.py` to test the model
- Set up **GitHub Actions** so that:
  - When code is pushed, it trains the model
  - Builds a Docker image
  - Tests the image automatically
  - Pushes it to DockerHub

### 3. Quantization (`quantization` branch)
- Loaded the trained model
- Compressed the model using **manual 8-bit quantization**
- Built the same model in **PyTorch**
- Tested the new model to check accuracy

## 📁 Files in This Project

- `train.py` → trains the model and saves it
- `predict.py` → loads the model and runs prediction
- `quantize.py` → compresses the model and tests it in PyTorch
- `Dockerfile` → builds a Docker container
- `.github/workflows/ci.yml` → GitHub Actions automation
- `requirements.txt` → list of required Python packages

## 📊 Final Model Comparison Table

| Metric       | Original Sklearn Model (`model.joblib`) | Quantized Model (`quant_params.joblib`) |
|--------------|-----------------------------------------|------------------------------------------|
| R² Score     | 0.6012                                  | 0.1657                                   |
| Model Size   | 414 Bytes                               | 469 Bytes                                |

