import os
import json
import numpy as np
from datetime import datetime

RESULTS_DIR = "results"
REPORT_PATH = os.path.join(RESULTS_DIR, "training_report.md")

# Load config
with open(os.path.join(RESULTS_DIR, "training_config.json")) as f:
    config = json.load(f)

# Load metrics
with open(os.path.join(RESULTS_DIR, "metrics_history.json")) as f:
    metrics = json.load(f)

# Load validation outputs
val_outputs = np.load(os.path.join(RESULTS_DIR, "val_outputs.npz"))
y_true = val_outputs["y_true"]
y_pred = val_outputs["y_pred"]

# Prepare metrics table
header = "| Epoch | Train Loss | Val Loss | AUC | Accuracy | Learning Rate |\n|-------|------------|----------|-----|----------|---------------|\n"
rows = ""
for m in metrics:
    rows += f"| {m['epoch']} | {m['train_loss']:.4f} | {m['val_loss']:.4f} | {m['auc']:.4f} | {m['accuracy']:.4f} | {m['learning_rate']:.2e} |\n"

# Report content
report = f"""
# 3D Prior Model Training Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Model Configuration

```
{json.dumps(config, indent=2)}
```

## Training and Validation Loss Curves

![Training Curves](training_curves_corrected.png)

## Metrics by Epoch

{header}{rows}

## Final Metrics

- **Best AUC:** {max(m['auc'] for m in metrics):.4f}
- **Best Accuracy:** {max(m['accuracy'] for m in metrics):.4f}
- **Final Train Loss:** {metrics[-1]['train_loss']:.4f}
- **Final Val Loss:** {metrics[-1]['val_loss']:.4f}

## Validation Outputs

- Saved as: `results/val_outputs.npz`

## All Results and Artifacts

- Training config: `results/training_config.json`
- Metrics history: `results/metrics_history.json`
- Training curves: `results/training_curves_corrected.png`
- Best model: `results/prior_3d_best_corrected.pth`
- Final model: `checkpoints/prior_3d_final_corrected.pth`

---

*Report generated automatically.*
"""

with open(REPORT_PATH, "w") as f:
    f.write(report)

print(f"Markdown report generated: {REPORT_PATH}") 