import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to PriceData.csv (root/data/ relative to this file's location)
_PART_B_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR   = os.path.dirname(_PART_B_DIR)
price_path  = os.path.join(_BASE_DIR, "data", "v2_PriceData.csv")

# v2 layout: col 0 = price_previous, cols 1-10 = hourly prices
# Load only cols 1-10 → shape (100, 10)
price_data = pd.read_csv(price_path).iloc[:, 1:].values  # shape (100, 10)

assert price_data.shape == (100, 10), f"Unexpected shape: {price_data.shape}"

# Average price per day: mean over 10 hours
avg_daily_price = price_data.mean(axis=1)  # shape (100,)

# Histogram
fig, ax = plt.subplots(figsize=(7, 4))

sns.histplot(avg_daily_price, bins=15, color="#4C72B0", edgecolor="black", linewidth=0.6, ax=ax)
ax.axvline(
    avg_daily_price.mean(),
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {avg_daily_price.mean():.3f} €/kWh",
)
ax.set_title("Distribution of Average Daily Energy Price (100 days)")
ax.set_xlabel("Average Price (€/kWh)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(_BASE_DIR, "part_B", "avg_daily_price_histogram.pdf"),
    bbox_inches="tight",
)
plt.show()

print(f"Mean avg daily price : {avg_daily_price.mean():.4f} €/kWh")
print(f"Std                  : {avg_daily_price.std():.4f} €/kWh")
print(f"Min / Max            : {avg_daily_price.min():.4f} / {avg_daily_price.max():.4f} €/kWh")
