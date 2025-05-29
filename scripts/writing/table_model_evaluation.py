import pandas as pd

# Path to CSV file
csv_path = "./results/eval/expansion_evaluation_summary.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Definitions for LaTeX table generation
field_map = {7: r"$\mathbb{F}_7$", 31: r"$\mathbb{F}_{31}$", 127: r"$\mathbb{F}_{127}$"}
latex_rows = []

# Build table data
for p in [7, 31, 127]:
    field_str = field_map[p]
    first_field_row = True
    for n in [3, 4, 5]:
        mid_n_row = False
        for k in [1, 3, 5]:
            mid_n_row = k == 3
            
            row = df[(df["p"] == p) & (df["n"] == n) & (df["k"] == k)]
            if row.empty:
                precision = recall = f1 = acc = "--"
            else:
                precision = f"{row.iloc[0]['precision'] * 100:.1f}"
                recall = f"{row.iloc[0]['recall'] * 100:.1f}"
                f1 = f"{row.iloc[0]['f1_score'] * 100:.1f}"
                acc = (
                    f"{row.iloc[0]['no_expansion_accuracy'] * 100:.1f}"
                    if pd.notna(row.iloc[0]['no_expansion_accuracy'])
                    else "--"
                )

            field_cell = field_str if first_field_row else ""
            var_cell = f"$n={n}$" if mid_n_row else ""
            latex_rows.append(
                f"      & {var_cell:<9} & {k} & {precision} & {recall} & {f1} & {acc} \\\\"
                if field_cell == ""
                else f"    \\multirow{{9}}{{*}}{{{field_cell}}}\n      & {var_cell:<9} & {k} & {precision} & {recall} & {f1} & {acc} \\\\"
            )
            first_field_row = False
            mid_n_row = False
        latex_rows.append("      \\cline{2-7}")

# Full LaTeX structure
latex_table = r"""\begin{table}[!t]
  \centering
    \caption{Evaluation results of Transformer predictions over polynomial rings $\mathbb{F}_p[x_1, \dots, x_n]$. Metrics are reported for different values of $k$.}
  \label{table:transformer-prediction}
  \begin{tabularx}{\linewidth}{l l c *{4}{Y}}
    \toprule
    Field & Variables & $k$ & Precision (\%) & Recall (\%) & F1 Score (\%) & No Expansion Acc. (\%) \\
    \Xhline{2\arrayrulewidth}
""" + "\n".join(latex_rows).rstrip(" \\") + r"""
    \bottomrule
  \end{tabularx}
\end{table}
"""

print(latex_table)

