# Dynamic RRG Charts with Plotly and DuckDB: A Powerful Combo for Relative Strength Analysis
## Visualizing Market Relative Strength with AI-Assisted Development

Have you ever watched sector rotation analyses on financial channels and wished you could apply that powerful insight to your own portfolio? Like many, I've seen the value of Relative Rotation Graphs (RRG) in understanding market dynamics, especially after watching some great content from StockCharts.com. The catch? Plotting your own RRG charts often requires a subscription to specialized platforms.

But what if I told you that with a little curiosity and the right tools, you could build your own 'institutional-grade' RRG chart plotter, completely customized to your needs, and accessible right from your browser? That's exactly what I set out to do, and the results were amazing! This project, initially sparked by a simple prompt to an AI, empowered me to create a dynamic RRG chart using open-source tools.

If you're interested in bypassing subscription fees, diving into the power of AI-assisted development, and gaining a powerful visualization tool for your sector rotation analysis, then keep reading. This post will guide you through building your very own RRG chart, just like I did.

---

### The Challenge & My AI-Assisted Approach

My journey began with a simple, yet ambitious, idea: could I leverage the power of generative AI to build this complex visualization tool? I decided to put ChatGPT to the test. My initial prompt was quite detailed, outlining all the core requirements for the function I envisioned:

```
Write a function to plot a RRG chart;

With trailing lines;

Benchmark: SPY

Other tickers: VTV,IYT,XBI,XLB,XLC,XLI,XLK,XLP,XLRE,XLU,XLY,XME;

Use plotly;

Render results in the browser;

Data source: duckdb file;

Table: daily_close;

Table structure daily_close(
  Date DATE,
  Ticker VARCHAR,
  Open DOUBLE,
  High DOUBLE,
  Low DOUBLE,
  "Close" DOUBLE,
  Volume BIGINT,
  PRIMARY KEY(Date, Ticker)
);

Accept input period for number of days, if nil input, use default 14;
```

The response I received from the AI was truly impressive, laying the groundwork for what you're about to build. This project became a blend of AI-generated code and my own refinements, demonstrating how powerful this collaborative approach can be.

---

### Setting Up Your Environment & Data Strategy

Before we dive into the code, let's make sure our environment is ready. You'll primarily be working with a Python script, and while I personally use VS Code (which you might also find helpful for its excellent Python support), it's completely optional. Any text editor will do!

**1. Python & Libraries:**
First things first, you'll need Python installed. Then, we'll install the necessary libraries:

```bash
pip install plotly pandas duckdb numpy yfinance
```

It's always a good practice to set up a virtual environment for your projects, but that's a topic for another day if you're just getting started.

**2. Your Data Source: DuckDB for Performance and Simplicity**
One of the core requirements I gave ChatGPT was to use a DuckDB file as the data source. I chose DuckDB for a few key reasons: its impressive performance, the ability to keep data tidier than a collection of separate Parquet files, and its excellent SQL support for querying data at various complexity levels.

You'll need a table to store your historical price data. I ended up using a table named `weekly_close` (though the initial prompt mentioned `daily_close`), with the following structure:

```sql
CREATE TABLE weekly_close (
  Date DATE,
  Ticker VARCHAR,
  Open DOUBLE,
  High DOUBLE,
  Low DOUBLE,
  "Close" DOUBLE,
  Volume BIGINT,
  PRIMARY KEY(Date, Ticker)
);
```

**Why weekly data?**
When using daily data, I observed that the RRG lines could wander quite messily throughout the four quadrants of the chart. Financial data is inherently noisy, and daily fluctuations can obscure the underlying trends. By aggregating to weekly data, the **lines became significantly smoother**, allowing for clearer identification of sector rotation and momentum shifts. This smoother movement helps in sensing the broader trends more effectively, which is what RRG charts are all about.

**Getting Your Data:**
While I have a custom data collection pipeline, you can easily get started. You can use the `yfinance` library to download daily historical data for your chosen tickers. Once you have daily data, you can aggregate it into weekly data using either Pandas or DuckDB's SQL capabilities. For example, you might select the last trading day of the week as your weekly close.

---

### Core Logic: RRG Calculation

This is where the magic happens! The heart of our RRG chart lies in calculating the Relative Strength (RS) Ratio and RS Momentum for each asset. While the exact formulas can vary slightly, my implementation uses a robust method to capture both the relative performance and its trend.

Here’s a breakdown of the Python code I used, which was largely informed by the AI's initial suggestions and my subsequent refinements:

```python
# Assuming 'df' contains the close prices for all tickers and benchmark
# and 'period' is our chosen lookback (e.g., 14 days/weeks)

# ---------------------------------------
# 1. Calculate Relative Strength (RS)
# ---------------------------------------
# Ratio of each ticker's close price to the benchmark's close price
rs = df[tickers].div(df[benchmark], axis=0)

# Apply Exponentially Weighted Moving Average (EWMA) to smooth the RS line
# This helps reduce noise and highlight trends in relative performance.
rs_ema = rs.ewm(span=period, adjust=False).mean()

# ---------------------------------------
# 2. Calculate RS-Ratio (X-axis of the RRG)
# ---------------------------------------
# The RS-Ratio measures the relative strength of the asset compared to its own average RS over the 'period'.
# It's scaled to center around 100 with a standard deviation scaling for consistency.
rs_ratio = 100 + (
    (rs_ema - rs_ema.rolling(period).mean()) /
    rs_ema.rolling(period).std()
)

# ---------------------------------------
# 3. Calculate RS-Momentum (Y-axis of the RRG)
# ---------------------------------------
# First, calculate the raw percentage change of the smoothed RS over the 'period'.
rs_mom_raw = rs_ema.pct_change(period)

# Similar to RS-Ratio, RS-Momentum is scaled to center around 100.
# It indicates whether the relative strength itself is improving or weakening.
rs_mom = 100 + (
    (rs_mom_raw - rs_mom_raw.rolling(period).mean()) /
    rs_mom_raw.rolling(period).std()
)

# ---------------------------------------
# 4. Combine Metrics & Calculate Rotation
# ---------------------------------------
data = {} # Dictionary to store all calculated metrics for each ticker

for ticker in tickers:
    # Combine the calculated RS-Ratio (x) and RS-Momentum (y) for each ticker
    combined = pd.concat(
        [rs_ratio[ticker], rs_mom[ticker]],
        axis=1
    ).dropna()
    combined.columns = ["x", "y"]

    # Calculate the angle of the ticker's position relative to the center (100, 100)
    # This helps in identifying the quadrant (Leading, Weakening, Lagging, Improving).
    dx = combined["x"] - 100
    dy = combined["y"] - 100
    angle = np.degrees(np.arctan2(dy, dx))
    angle = (angle + 360) % 360 # Normalize angle to 0-360 degrees

    # Calculate the 'velocity' or change in angle, indicating the rotation speed/direction.
    velocity = angle.diff()
    velocity = (velocity + 180) % 360 - 180 # Normalize velocity to -180 to 180

    combined["angle"] = angle
    combined["velocity"] = velocity

    data[ticker] = combined
```

**Understanding the Quadrants:**
The center of our RRG chart is at (100, 100).
* **Leading Quadrant (Top Right):** RS-Ratio > 100, RS-Momentum > 100. Strong relative performance, and that strength is improving.
* **Weakening Quadrant (Bottom Right):** RS-Ratio > 100, RS-Momentum < 100. Still strong relative performance, but its momentum is declining.
* **Lagging Quadrant (Bottom Left):** RS-Ratio < 100, RS-Momentum < 100. Weak relative performance, and that weakness is increasing.
* **Improving Quadrant (Top Left):** RS-Ratio < 100, RS-Momentum > 100. Weak relative performance, but its momentum is improving.

**Dynamic Symmetric Scaling for Optimal View:**
One crucial refinement I added was dynamic symmetric scaling. Initially, when all the lines cluster around the center, the chart can look messy and hard to read. To combat this, the plotting function dynamically adjusts the chart's axes to ensure all trailing lines and current points are visible, while keeping the center at (100, 100) and maintaining a square aspect ratio. This ensures we utilize the chart area effectively without distorting the visual interpretation of the RRG.

---

### Plotting with Plotly: Bringing the RRG to Life

With our `rs_ratio` (X-axis) and `rs_momentum` (Y-axis) calculated, the next step is to visualize this data. This is where Plotly truly shines. Its interactive capabilities transform a static chart into a dynamic analysis tool, allowing for deeper exploration of sector rotation over time.

Beyond just drawing points and lines, I leveraged Plotly's animation and slider features to provide an experience akin to what you'd find in professional platforms.

**Key Plotly Features Utilized:**

* **`go.Figure()` and `go.Scatter()`:** These are the building blocks for creating our scatter plot. Each ticker's trajectory (the trailing line) and its current position are plotted using `go.Scatter` traces.
* **Trailing Lines & Current Points:** Each ticker gets two components: a subtle `lines` trace showing its historical path on the RRG, and a `markers+text` trace for its most current position, clearly labeled with the ticker symbol.
* **Hover Information:** Detailed hover text for each ticker shows its current `rs_ratio`, `rs_momentum`, and the corresponding date. This is crucial for precise analysis.
* **Dynamic Quadrants:** The RRG chart is divided into four quadrants (Leading, Weakening, Lagging, Improving). Plotly's `add_shape` function is used to visually represent these quadrants, making it immediately clear where each asset stands.
* **Interactive Play, Pause, and Slider Controls:** This is perhaps the most exciting part! Plotly allows you to animate the chart over time. This means you can:
    * **Play:** Watch the RRG lines move and evolve, observing the rotation of sectors as if you're watching a real-time simulation.
    * **Pause:** Stop the animation at any point to analyze the chart's state on a specific date.
    * **Slider:** Manually drag a slider across the historical timeline to scrub through the data, allowing for detailed observation of movements day-by-day (or week-by-week in our case). This is incredibly handy for understanding the subtle shifts in momentum and relative strength.
* **Toggle Ticker Visibility:** Plotly automatically generates a legend where you can click on any ticker to hide or show its line and current point. This is super useful for decluttering the chart and focusing on specific assets you're interested in.

By combining these features, we don't just get a plot; we get an interactive RRG dashboard that helps us truly sense the rotation patterns in the market.

---

### Code

Here's the complete Python script for generating your dynamic RRG chart. (Remember to add the `argparse` or `sys.argv` logic for handling command-line arguments, as discussed, to make it runnable.)

```python
# >>> PASTE YOUR COMPLETE PYTHON SCRIPT HERE <<<
# This should include your function definition, RRG calculation logic,
# Plotly plotting, and the command-line argument parsing.

# Example for argument parsing (you will need to implement this in your script):
# import argparse
#
# if __name__ == "__main__":
# parser = argparse.ArgumentParser(description="Plot a dynamic RRG chart.")
# parser.add_argument("duckdb_file_path", type=str, help="Path to the DuckDB database file.")
# parser.add_argument("rrg_period", type=int, help="Number of periods for RRG calculation (e.g., 14).")
# parser.add_argument("trailing_points", type=int, help="Number of historical points to show for trailing lines.")
# parser.add_argument("tickers", type=str, help="Comma-separated list of tickers (e.g., 'SPY,VTV,XBI').")
#
# args = parser.parse_args()
#
# tickers_list = [t.strip() for t in args.tickers.split(',')]
#
# # Call your main plotting function here with args.duckdb_file_path, args.rrg_period, etc.
# # plot_rrg_chart(args.duckdb_file_path, 'SPY', tickers_list, args.rrg_period, args.trailing_points)
```

---

### Usage & How to Run Your RRG Chart

Now that we've covered the setup and the logic, let's get your RRG chart up and running! My goal was to make this script as easy to use as possible, so you don't need to dive into the code itself to change parameters. You can simply run it from your command prompt or terminal.

**1. Save the Script:**
First, save your Python code (including the `plot_rrg_chart` function and the logic to parse command-line arguments) into a file, for example, `plot_rrg_chart.py`.

**2. Prepare Your DuckDB File:**
Ensure you have your `weekly_close.duckdb` (or whatever you named your DuckDB file) ready with the historical data. Remember, you can use `yfinance` to fetch data and then populate this DuckDB file.

**3. Run from the Command Line:**
Open your terminal or command prompt, navigate to the directory where you saved your `plot_rrg_chart.py` file, and execute the script using the following format:

```bash
python plot_rrg_chart.py <duckdb_file_path> <rrg_period> <trailing_points> "<tickers_comma_separated>"
```

Let's break down the arguments:
* `<duckdb_file_path>`: The full path to your DuckDB database file (e.g., `C:\temp\my_data.duckdb` or `/users/yourname/data/weekly_close.duckdb`).
* `<rrg_period>`: The number of days/weeks to use for the RRG calculation (e.g., `14` for a 14-period RRG).
* `<trailing_points>`: The number of historical points to show for each ticker's trailing line. This controls how much of the past trajectory is visible.
* `"<tickers_comma_separated>"`: A **quoted**, comma-separated list of the ticker symbols you want to plot. Make sure to include your benchmark ticker here if you want to explicitly see its (stationary) position. For example, `"SPY,VTV,IYT,XBI"`.

**Example Execution:**
Here’s an example using the parameters we discussed earlier:

```bash
python plot_rrg_chart.py C:\temp\my_financial_data.duckdb 14 7 "SPY,VTV,IYT,XBI,XLB,XLC,XLI,XLK,XLP,XLRE,XLU,XLY,XME"
```

Upon execution, a new browser window will open, displaying your interactive RRG chart with all the dynamic features we discussed!

---

### Screenshots & Video Demo

Here are some visuals of the RRG chart in action, showcasing its interactive features and how it helps visualize sector rotation.

**Static Screenshot of the RRG Chart:**
*(Insert your high-quality screenshot here, perhaps with annotations highlighting the quadrants or key features.)*
![RRG Chart Screenshot](link_to_your_screenshot.png)

**Interactive Demo Video:**
*(Insert a link to your video demo, showing the play/pause, slider, and toggle features.)*
[![RRG Chart Video Demo](link_to_your_video_thumbnail.jpg)](link_to_your_video.mp4)

---

### Potential Enhancements & Future Work

This project is a solid foundation, but there's always room to grow! Here are a few ideas for future enhancements:

* **Different Smoothing Periods:** Experiment with varying lookback periods for the RS-Ratio and RS-Momentum to observe different sensitivities to market changes.
* **More Advanced Data Sources:** Integrate with real-time data feeds or other financial APIs.
* **Web Application Integration:** Build a simple web interface using frameworks like Dash to make the chart accessible via a browser without needing to run the script locally.
* **Sophisticated RRG Calculations:** Explore alternative RRG calculation methodologies, perhaps closer to J. M. van Vliet's original work, for comparative analysis.
* **Custom Benchmarks:** Allow users to specify any ticker as a benchmark, or even a custom index.

---

### Conclusion & Your Next Steps

You've now got the tools and the knowledge to generate your own dynamic Relative Rotation Graphs! While we used `SPY` as our benchmark for this example, which is often considered a 'golden standard' due to its broad market representation, feel free to experiment. I've even tried equal-weighted benchmarks, but honestly, I found the movements and comparisons far clearer and more intuitive when benchmarked against `SPY`.

This project really showed me the power of AI-assisted development. With a clear prompt and a bit of iteration, you can turn complex analytical desires into functional, 'institutional-grade' tools.

So, what are you waiting for? **Try it, run it, and dive into your market analysis!** I'm confident you'll quickly come to love this animated RRG chart as a powerful, customizable trading research tool. Watch how sectors rotate, identify trends before they're headline news, and make more informed decisions. The ability to visualize these dynamics interactively truly changes the game for understanding market cycles. Happy charting!
