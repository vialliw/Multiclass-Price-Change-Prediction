# Relative Rotation Analysis Dashboard

<p align="center">
<video src="https://github.com/user-attachments/assets/27def6b3-4c2f-43ef-9d2f-bc21bcec12d4?raw=true" width="900" autoplay loop muted playsinline></video>
</p>

## Abstract

Traditional performance metrics often fail to capture the interplay between trend strength and momentum. This project provides a **Relative Rotation Analysis (RRA)** engine that normalizes asset performance against a benchmark, allowing traders to visualize lead-lag relationships and anticipate sector rotation before it appears in absolute price charts.

## Core Methodology

The dashboard utilizes a two-factor quantitative model centered on a **100-baseline**:

### 1. RS-Ratio (X-Axis)

Measures the relative trend of an asset. It is derived from the Rate of Change (ROC) of the price ratio between the asset and the benchmark:

<p align="left">
<img src="https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/images/rs.ratio.svg" width="450">
</p>

### 2. RS-Momentum (Y-Axis)

Measures the acceleration of the relative trend. It identifies whether the relative strength is gaining or losing velocity:

<p align="left">
<img src="https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/images/rs.momentum.svg" width="450">
</p>

---

## Technical Features

* **Dynamic Scaling Engine:** Automatically adjusts axis limits to ensure 90% chart utilization, preventing historical outliers from squashing current data.
* **Vector Smoothing:** Implements `scipy.interpolate.make_interp_spline` for smooth trailing lines, making directional shifts easier to identify visually.
* **High-Performance Backend:** Integrated with **DuckDB** for rapid analytical queries on Parquet and large-scale financial datasets.
* **High-Fidelity Visual Mapping:**
* **13-Color Chromatic Spectrum:** Implements a discrete palette (Red to Magenta) ensuring maximum contrast between overlapping asset trails.
* **Deterministic Trace Identity:** Legend synchronization via `legendgroup` allows simultaneous toggling of asset heads and trailing vectors while maintaining persistent color identity.



---

## Quick Start

### Usage Implementation

The following snippet highlights the localized "trailing" logic and dynamic coordinate scaling used in the dashboard:

```python
# Slicing logic for strict trail length control
for i in range(trail-1, len(dates)):
    start_idx = max(0, i - trail + 1)
    tail_slice = slice(start_idx, i + 1)
    
    # Generate smoothed spline for the trail
    xr, yr = clean_ratio[ticker].iloc[tail_slice], clean_momo[ticker].iloc[tail_slice]
    xs, ys = get_spline(xr.values, yr.values)
    
    # Re-stating width inside frame to ensure visual consistency
    frame_data.append(go.Scatter(x=xs, y=ys, line=dict(width=3, color=colors[idx])))

```

---

## Planned Improvements & Roadmap

* **Dynamic Benchmark Injection:** Update the core execution script to accept the `benchmark` ticker as a command-line argument. This will allow users to seamlessly pivot the analysis between different asset classes:
* **Equities:** Benchmark against `$SPY` or `$QQQ`.
* **Commodities:** Benchmark against broad indexes like `$DBC` or specific sector leads like `$GLD`.


* **Volatility-Adjusted Momentum:** Integration of ATR-based normalization for RS-Momentum to better handle high-volatility commodity markets.

---

## License & Trademark Notice

* **Software License:** MIT License.
* **Trademark Notice:** "RRG" and "Relative Rotation Graphs" are registered trademarks of [RRG Research](https://www.relativerotationgraphs.com/). This project is an independent quantitative implementation of rotation analysis concepts and is not affiliated with, endorsed by, or sponsored by RRG Research.

---
