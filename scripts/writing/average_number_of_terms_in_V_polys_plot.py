import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("results/eval/average_number_of_terms_in_V_polys_num_samples=1000.csv")

# Setup plot
fig, ax = plt.subplots(figsize=(8, 5))  # <- Adjust figure size here

p_list = [7, 31, 127]
colors = ['tab:blue', 'tab:orange', 'tab:green']  # <- Change colors here
linestyles = ['-', '--', '-.']                    # <- Change line styles here

for i, p in enumerate(p_list):
    subset = df[df['p'] == p]
    x = subset['n']
    y = subset['average_support_sizes']
    ax.plot(
        x, y,
        marker='o',
        color=colors[i],
        linestyle=linestyles[i],  # <- Line style
        linewidth=2.0,            # <- Line width
        label=f"$\\mathbb{{F}}_{{{p}}}$"
    )

# Axes, title and labels
ax.set_xlabel("Number of Variables (n)", fontsize=12)
ax.set_ylabel("Average Number of Terms", fontsize=12)
ax.set_title("Average Support Sizes in $\\mathbb{F}_p[x_1, ..., x_n]$", fontsize=14)

# Legend and grid
ax.legend(title="Field", fontsize=10, title_fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig("results/eval/average_number_of_terms_in_V_polys_num_samples=1000.pdf")
