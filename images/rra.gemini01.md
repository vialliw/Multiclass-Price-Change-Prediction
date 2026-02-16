# Relative Rotation Analysis Dashboard

![Stock Price Change Prediction Using Random Forests](https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/images/stock_price_change_prediction_using_random_forests.jpg?raw=true)


An institutional-grade visualization engine for cross-asset cyclical analysis. This dashboard implements relative strength and momentum vectors to track the rotation of financial instruments (Equities, Commodities, and FX) across four distinct market quadrants.

## Abstract

Traditional performance metrics often fail to capture the interplay between trend strength and momentum. This project provides a **Relative Rotation Analysis (RRA)** dashboard that normalizes asset performance against a benchmark, allowing traders to visualize lead-lag relationships and anticipate sector rotation before it appears in absolute price charts.

## Core Methodology

The dashboard utilizes a two-factor quantitative model centered on **100**:

### 1. RS-Ratio (X-Axis)
Measures the relative trend of an asset. It is derived from the Rate of Change (ROC) of the price ratio between the asset and the benchmark:

<p align="center">
  <img src="https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/images/rs.ratio.svg" width="500">
</p>

### 2. RS-Momentum (Y-Axis)
Measures the acceleration of the relative trend. It identifies whether the relative strength is gaining or losing velocity:

<p align="center">
  <img src="https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/images/rs.momentum.svg" width="500">
</p>

---

## Technical Features

* **Dynamic Scaling Engine:** Automatically adjusts axis limits to ensure 90% chart utilization, preventing historical outliers from squashing current data.
* **Vector Smoothing:** Implements `scipy.interpolate.make_interp_spline` for smooth trailing lines, making directional shifts easier to identify visually.
* **High-Performance Backend:** Integrated with **DuckDB** for rapid analytical queries on Parquet and large-scale financial datasets.
* **Interactive Controls:** Full animation suite with localized "trailing" logic to visualize path dependency over time.
* **Optimized Chromatic Spectrum:** Implements a discrete 13-color palette mapped across the visible spectrum (Red to Magenta). This ensures maximum contrast between overlapping asset trails, critical for distinguishing between high-correlation sectors (e.g., XLK vs. XLC).
* **Deterministic Trace Identity:** Trace colors are bound to specific ticker objects through a legendgroup synchronization layer. This allows the user to toggle an asset's visibility in the animation frames while maintaining persistent color identity across the Ratio and Momentum heads.
---

## Quick Start

### Prerequisites

```bash
pip install duckdb pandas numpy plotly scipy

```

### Usage Implementation

The following snippet demonstrates the core animation and scaling logic used to generate the dashboard:

```python
# Clip from plot_rrg_chart_gemini09d.py
# Implementation of Dynamic Scaling and Frame Animation

# Calculate limits based on visible data for 90% utilization
all_visible_vals = np.concatenate([clean_ratio.values, clean_momo.values])
max_dev = max(abs(all_visible_vals.max() - 100), abs(all_visible_vals.min() - 100))
limit_half = max_dev / 0.9
limit = [100 - limit_half, 100 + limit_half]

# Build Animation Frames with Strict Trail Length
for i in range(trail-1, len(dates)):
    start_idx = max(0, i - trail + 1)
    tail_slice = slice(start_idx, i + 1)
    
    # Generate smoothed spline for the trail
    xr, yr = clean_ratio[ticker].iloc[tail_slice], clean_momo[ticker].iloc[tail_slice]
    xs, ys = get_spline(xr.values, yr.values)
    
    # Update frame data...

```

---

## Performance Interpretation

| Quadrant | RS-Ratio | RS-Momentum | Context |
| --- | --- | --- | --- |
| **LEADING** | > 100 | > 100 | Strong trend and accelerating momentum. |
| **WEAKENING** | > 100 | < 100 | Trend remains positive but momentum is fading. |
| **LAGGING** | < 100 | < 100 | Negative relative trend and declining momentum. |
| **IMPROVING** | < 100 | > 100 | Trend is negative but momentum is shifting upward. |

---

## License & Trademark Notice

* **Software License:** MIT License.
* **Trademark Notice:** "RRG" and "Relative Rotation Graphs" are registered trademarks of [RRG Research](https://www.relativerotationgraphs.com/). This project is an independent quantitative implementation of rotation analysis concepts and is not affiliated with, endorsed by, or sponsored by RRG Research.

---

### Suggested Next Steps for your Repo:

1. **Add a `docs/` folder:** Place a few exported `.html` files or `.png` screenshots of the dashboard in action.
2. **Benchmark Flexibility:** Update your script to take the `benchmark` ticker as a command-line argument so users can easily switch between  (Stocks) and  (Commodities).
