import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# === Load Data ===
csv_path = "results/eval/back_transform_analysis_num_samples=1000.csv"  # <- Change CSV filename as needed
df = pd.read_csv(csv_path, header=None)

# === Format Headers ===
x_values = df.iloc[0, 2:].astype(float).values  # x-axis values (3, 4, ..., 10)
data = df.iloc[1:].copy()
data.columns = ['p', 'n'] + list(x_values)
data = data.drop(columns=[col for col in data.columns if col not in ['p', 'n'] + list(x_values)])

# === Prepare Plot ===
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # <- Adjust figure size here
n_list = [3, 4, 5]
p_list = [7, 31, 127]
colors = ['tab:blue', 'tab:orange', 'tab:green']  # <- Change line colors here
linestyles = ['-', '--', '-.']  # <- Change line styles here (in same order)

for i, n in enumerate(n_list):
    ax = axs[i]
    for j, p in enumerate(p_list):
        subset = data[(data['p'] == str(p)) & (data['n'] == str(n))]
        if not subset.empty:
            y = subset.iloc[0, 2:].astype(float).values
            ax.plot(
                x_values,
                y,
                marker='o',                    # <- Can change marker type (e.g. 's', '^', 'x')
                label=rf"$\mathbb{{F}}_{{{p}}}$",  # <- LaTeX notation
                color=colors[j],              # Color
                linestyle=linestyles[j],      # Style
                linewidth=4.0,                 # <- Adjust line width
                alpha=0.8
            )

    ax.set_title(rf"$n={n}$", fontsize=28)       # <- Title font size
    ax.set_xlabel(rf"Size of $F$", fontsize=24)  # <- x-axis label
    ax.tick_params(axis='both', labelsize=20)   # <- Adjust axis number size
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # <- Set x-axis to integer ticks
    if i == 0:
        ax.set_ylabel("Success Rate", fontsize=24)

# === Legend Settings ===
axs[-1].legend(fontsize=24, loc='lower right')  # <- Legend font size and position

# === Finalize ===
plt.tight_layout()
# plt.show()
plt.savefig(f"results/eval/back_transform_analysis_num_samples=1000.pdf")