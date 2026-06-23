# Nifty 50 Multimodal Directional Prediction
### MSc Machine Learning Thesis — Liverpool John Moores University

A multimodal deep learning framework for predicting the directional 
movement of the Nifty 50 index by integrating quantitative technical 
indicators with LLM-generated sentiment from financial news.

## Key Result
**63.66% directional accuracy** on 432 out-of-sample test days  
(June 2024 – March 2026) — +9.72pp over technical-only baseline

## Framework Architecture
- **Numerical branch:** 4-block dilated TCN (32 channels, GELU, dropout 0.3)
- **Text branch:** GPT-4o few-shot summarisation → FinBERT 773-dim embeddings
- **Fusion:** Concatenation → Deep FNN (256→128→64→32→1)
- **Loss:** Huber Loss (δ=0.5)

## Pipeline Overview

| Step | Description |
|------|-------------|
| Step 1 | Nifty 50 OHLC data collection (Jan 2017 – Mar 2026) |
| Step 2 | Technical indicator computation (EMA, RSI, BB, MACD) |
| Step 3 | Log-return transformation + ADF stationarity testing |
| Step 4 | StandardScaler preprocessing + train/test split |
| Step 5 | GDELT news extraction (81,936 articles, 10 topics) |
| Step 6 | GPT-4o few-shot summarisation (2,817 daily summaries) |
| Step 7a | FinBERT sentiment scoring |
| Step 7b | FinBERT 768-dim embedding generation |
| Step 8 | Dataset preparation and alignment |
| Step 9 | Model training and evaluation |

## Results Summary

| Model | Dataset | Directional Accuracy | R² |
|-------|---------|---------------------|-----|
| Naive baseline | — | ~51.00% | — |
| Baseline LSTM (OHLC only) | 11yr | 53.94% | -0.001 |
| Linear Transformer + FinBERT | 3yr | 57.35% | -0.025 |
| TCN + Deep FNN | 3yr | 58.09% | -0.025 |
| TCN + Cross-Attention | 3yr | 52.94% | 0.025 |
| TCN + Rolling Sentiment | 3yr | 52.94% | — |
| Dual TCN | 3yr | 50.74% | 0.022 |
| **TCN + Deep FNN (FINAL)** | **9yr** | **63.66%** | **0.154** |

## Data Sources
- **Price data:** NSE India / Yahoo Finance (^NSEI)
- **News data:** GDELT Project (DOC API + Full Text Search API)
- **LLM summarisation:** OpenAI GPT-4o API
- **Sentiment embedding:** FinBERT (ProsusAI/finbert via Hugging Face)

## Requirements

torch
transformers
pandas
numpy
scikit-learn
openai
requests
