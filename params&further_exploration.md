# Tasks
3 compare

# Dates
## Train and validation
start_date = datetime(2025, 11, 4, 19, 30, 0, tzinfo=timezone.utc)
end_date = datetime(2025, 5, 30, 19, 30, 0, tzinfo=timezone.utc)

## Embargo
start_date = datetime(2025, 5, 30, 19, 30, 0, tzinfo=timezone.utc)
end_date = datetime(2026, 6, 1, 3, 30, 0, tzinfo=timezone.utc)

## Test
start_date = datetime(2026, 6, 1, 3, 30, 0, tzinfo=timezone.utc)
end_date = datetime(2026, 6, 21, 23, 0, 0, tzinfo=timezone.utc)


# Showcases
Interactive Sweep Visualization HTML saved to: /home/alfred/github/fts/runs/reports/sweep_report_lstm_hidden_dim_sweep.html
Interactive Sweep Visualization HTML saved to: /home/alfred/github/fts/runs/reports/sweep_report_lstm_num_layers_sweep.html


# Questions
1. Is my RNN teacher forcing? That is, does the model see y(t-1) at t?
2. Why should I show slice plots of unchanged hyperparams? Only show those that have more than a singleton value.

# Case study:
Model 48 vs model 29
LSTM models
Both trained on the same period:

Model 29 had just had log returns for close
Then I added the feature pipeline, which includes this and other features. Model 48 had 5 featres. 
Performance was worse across IC, 

How can this be?

  This is a very common phenomenon in quantitative research when moving from a single feature to multi-dimensional feature spaces. Here are the core reasons why your 1D model (Model 29) is            
  outperforming the 5D pipeline model (Model 48) on validation metrics:
  ──────
  ### 1. The Signal-to-Noise Ratio (SNR) Dilution
  
  • The close log return is the target variable. It has the highest direct autocorrelation and holds the strongest linear signal for predicting the next close return.
  • Volume Change (ln (Volumeₜ/Volumeₜ₋₁)) and Intra-bar ratios (like High/Close or Low/Close) are notoriously noisy in high-frequency/crypto data (30m BTC).
  • When you add these 4 extra features, you increase the noise of the input space. If the LSTM's weights are not heavily regularized, it will start fitting to spurious noise patterns in the volume or
  intra-bar tails, diluting the clean predictive signal from the close price returns.
  
  ### 2. Wildly Mismatched Distributions (Even After Scaling)
  
  Even though the trainer applies  StandardScaler  (z-score normalization), the underlying distributions of these features are very different:
  
  • Close returns are symmetric, near-zero mean, and relatively bounded.
  • High/Close ratio is strictly positive (≥0).
  • Low/Close ratio is strictly negative (≤0).
  • Volume change is extremely fat-tailed with huge spike outliers (e.g. volume doubling or dropping to 10% in a single bar).
  LSTMs are sensitive to extreme outliers and asymmetric distributions. An extreme outlier in volume return can cause massive gradient updates that destabilize the learned LSTM weights for the price  
  return features.
  
  ### 3. Curse of Dimensionality & Overfitting
  
  By moving from 1 feature to 5 features:
  
  • The dimension of the input layer of the LSTM went from  1  to  5 .
  • This increases the parameter count of the first LSTM layer.
  • With a small dataset (e.g. 5000–7000 training samples), this extra capacity makes the model prone to overfitting on the training set, causing the validation loss to degrade and the validation     
  Information Coefficient (IC) to drop from  0.137  to  0.107 .
  
  ### 4. Hyperparameter Suboptimality
  
  The neural network hyperparameters (hidden dimensions, learning rate, dropout, weight decay) in your search space were likely tuned or initialized to perform well on 1D close returns.
  
  • A model architecture that is optimal for a 1D sequence is often too simple or too complex for a 5D sequence.
  • For the 5D pipeline model to shine, you typically need to re-run the Optuna search with higher dropout rates, stronger L2 weight decay (regularization), or a smaller hidden dimension to force the 
  network to ignore the noise and focus only on the strong signals.



# Further 
## modifications
Incorporating to training:
1. Slippage
2. Time delay
3. Measures of latency and non-fills

## study:
1. If I will set up the backtest to have an execution delay of 1 bar, wouldn't it be good to set up as training target the 2-bar return?

2. Pronostico de vol? No hay buenos productos rápidos? 

3. Portfolio of models both predicting on the same bar 
class PortfolioStrategy
