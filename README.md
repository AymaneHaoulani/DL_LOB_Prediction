# Predicting Bitcoin Mid-Price Trends Using Deep Learning Models

## Ethical Disclaimer
This study is designed for research purposes only and should not be used as a financial advisory tool. Cryptocurrency markets are highly volatile, and the models predictions are subject to errors due to market anomalies, poor input data, or systemic issues. Over-reliance on AI models for financial decision-making can result in significant losses. Users should independently verify predictions and exercise caution when using this tool for trading or investment purposes.

## Overview
This repository contains the code and resources for our study on predicting Bitcoin mid-price trends using Limit Order Book (LOB) data. We evaluate multiple machine learning and deep learning models, including Logistic Regression, LSTM, CNN-LSTM, and transformer-based architectures. Our proposed model, **BiTran**, introduces novel enhancements such as Time Absolute Position Encoding (tAPE) and optimized attention mechanisms, achieving superior performance for trend forecasting over various time horizons.

## Recommendation
We highly recommend using Google Colab Pro for this project, as it provides access to powerful GPUs and increased memory, which are essential for handling the complex architectures and the enormous dataset (~8 million data points) used in this work. Additionally, pretrained models are included in the pretrained path, allowing users to leverage prebuilt weights for faster experimentation and evaluation without the need for extensive training from scratch.

## Key Features

- **Data Collection**: Real-time Bitcoin LOB data collected via Binance's WebSocket API.
- **Feature Engineering**: Econometric and statistical features designed for LOB data.
- **Model Comparison**: Extensive evaluation of Logistic Regression, LSTM, CNN-LSTM, and transformers.
- **BiTran Architecture**: A novel transformer-based model integrating:
  - Time Absolute Position Encoding (tAPE) for temporal relationships.
  - Enhanced attention mechanisms for better long-term trend prediction.
- **Robust Testing**: Evaluation over multiple prediction horizons (10, 20, 50, and 100 timesteps).

## Results
Detailed results for each model are provided in `results/` and our accompanying research paper.

## Repository Structure

```plaintext
├── Models/               # Implementation of all models.
│   ├── Logistic Regression/
│   ├── CNN-LSTM/
│   ├── LSMT/
│   ├── Transformers/
├── src/                
│   ├── Evaluation/
│   ├── Preprocessing/
│   ├── Training/
├── Pretrained/            # Pretrained Models.
│   ├── BiTran_k.pt
│   ├── pretrained_example.py
├── Results/              # Output logs, model checkpoints, and performance metrics.
│   ├── CNN-LSTM/
│       ├── CNN-LSTM_Results.ipynb
│   ├── LSTM/
│       ├── LSTM_Results.ipynb
│   ├── Transformers/
│       ├── Transformers_Results.ipynb
│   ├── Logistic Regression
├── README.md             # Project documentation.
├── requirements.txt      # Python dependencies.
```
