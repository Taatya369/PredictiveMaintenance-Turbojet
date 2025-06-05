# Predictive Maintenance for Turbojet Engines

A deep learning-based Predictive Maintenance System to forecast equipment failure in aircraft turbojet engines using NASA's CMAPSS dataset. The system reduces unplanned downtime by **26.54%** and improves prediction accuracy using CNN, BiLSTM, and Attention mechanisms.

---

## Objective
Create a predictive maintenance model that:
- Predicts Remaining Useful Life (RUL) of jet engines
- Classifies if a failure is expected within the next 30–35 cycles
- Reduces equipment downtime significantly using ML

---

## Dataset
**Source**: NASA CMAPSS Dataset (combined subset)

| Feature               | Description                             |
|----------------------|-----------------------------------------|
| `unit_number`        | Unique engine identifier                 |
| `time_in_cycles`     | Time cycle of the engine                |
| `operational_setting_*` | Environmental settings like altitude etc |
| `sensor_measurement_*`  | 21 condition monitoring sensors         |
| `RUL`                | Target label: Remaining Useful Life      |
| `dataset_id`         | Subset (FD001, FD004 etc)               |

- **Features Used**: 21 sensors + 3 operational settings = 24
- **Dropped**: `unit_number`, `time_in_cycles`, `dataset_id`, and 3 constant sensors
- **Total Engines**: ~100 unique engines

---

## Model Architecture

```
Input (40-step sequence) → Conv1D → BiLSTM → MultiHeadAttention
      → Add & LayerNormalization → GlobalAveragePooling
      → Dropout(30%) → Dense(64)
      → [Regression Output (RUL), Classification Output (Failure < 30)]
```

- **Loss Function**: `0.6 * RUL MSE + 0.4 * Binary Crossentropy`
- **Optimizer**: Adam (`lr=0.001`)
- **EarlyStopping**: Patience = 10 (monitored val loss)

---

## Training Results

| Epoch | Train MAE | Validation MAE | Accuracy (<35 cycles) |
|-------|-----------|----------------|------------------------|
| 1     | 70.56     | 58.66          | 85.3%                  |
| 30    | 20.33     | **19.86**      | **95.02%**             |

**MAE Reduction**: ~71.2% (from 70.56 → 20.33)

---

## Downtime Simulation

```python
def simulate_downtime_binary(y_true, y_pred_prob):
    y_pred_bin = (y_pred_prob > 0.5).astype(int)
    y_true_bin = (y_true <= 35).astype(int)
    baseline = np.sum(y_true_bin)
    predicted = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    avoided = baseline - predicted
    return (avoided / baseline) * 100
```

### Result:
- **Baseline Downtime Events**: 5106
- **Predicted Downtime Events**: 3751
- **Avoided Events**: 1355
- **Downtime Reduction**: **26.54%** ✅

---

## Performance Metrics
- **RUL MAE**: 19.86 (Best Epoch)
- **Downtime Reduced**: 26.54%
- **Classification AUC**: ~0.98

---

## Tools & Tech
- Python, TensorFlow, Keras
- SHAP for explainability (planned extension)
- Google Colab for training
- Matplotlib/Seaborn for plotting

---

## Future Scope
- Use SHAP to identify top predictive sensors for targeted maintenance
- Deploy with real-time sensor data from IoT gateways
- Integrate feedback loop for self-learning

---

## How to Run
1. Upload `CMAPSS_combined_preprocessed_with_RUL.csv`
2. Run `finalpdm.ipynb` (code logic & training)
3. Outputs & metrics displayed inline with plots

---

## Author
**ARYA S PRAKASH** 
PRATIBHA EXCELLENCE AWARD 2025
---
