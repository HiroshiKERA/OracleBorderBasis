#!/usr/bin/env python3

import argparse
from pathlib import Path
import itertools as it
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from src.dataset.generators.expansion_generator import ExpansionGenerator, compute_border_basis_with_timeout
from src.border_basis_lib.improved_border_basis import ImprovedBorderBasisCalculator
from src.loader.data_format.processors.expansion import ExtractKLeadingTermsProcessor
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus
from src.dataset.processors.utils import poly_to_sequence

from sage.all import *


def get_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel"
    )
    parser.add_argument(
        "--k_last_calls",
        type=int,
        default=5,
        help="Number of last calls to use"
    )
    parser.add_argument(
        "--from_csv",
        action="store_true",
        help="Only re-plot from CSV"
    )
    return parser

def list_paths(ps=[7, 31, 127], ns=[3, 4, 5], base='./data/border_basis', skip_patterns=None):
    base = Path(base)
    for p, n in it.product(ps, ns):
        if (p, n) in skip_patterns:
            continue
        
        comb = f"p={p}_n={n}"
        degree_bounds = '_'.join([str(4)] * n)
        dirname = Path(f'GF{p}_n={n}_deg=1_terms=10_bounds={degree_bounds}_total={2}')
        data_path = base / dirname / 'test.infix'

        yield {'tag': (p, n), 'data_path': data_path}


def get_leading_terms_processor(k, separator=' [SEP] ', supersparator=' [BIGSEP] '):
    return ExtractKLeadingTermsProcessor(k, separator=separator, supersparator=supersparator)

def get_monomial_processor(ring):
    monomial_processor = MonomialProcessorPlus(
                num_variables=ring.ngens(),
                max_degree=20,
                max_coef=int(ring.base_ring().order())  # 'GF7' -> 7
            )
    return monomial_processor
    

def _run(F, ring, k_last_calls=5):    
    calculator = ImprovedBorderBasisCalculator(ring, corollary_23=True, N=100, save_universes=True, verbose=False, sorted_V=True)
    try:
        G, O, timings = compute_border_basis_with_timeout(
            calculator, 
            F,
            use_fast_elimination=True,
            lstabilization_only=False
        )
    except TimeoutError:
        print(f"Timeout occurred for sample, skipping...")
        return [], {"timeout": True}
    
    # Get the last L-stable span call
    last_Lstable_span_dataset = calculator.datasets[-1]
    if k_last_calls:
        last_Lstable_span_dataset = last_Lstable_span_dataset[-k_last_calls:]
    
    samples = [(sample[0], sample[1]) for sample in last_Lstable_span_dataset]
    return samples


def run(args, config_path, source_data_path, n_jobs=1, k_last_calls=5):
    k_last_calls = 5 
    DatasetGenerator = ExpansionGenerator
    generator = DatasetGenerator.from_yaml(config_path, None)
    ring = generator.ring
    Fs = generator.load_and_process(f'{source_data_path}/test.infix')
    Fs = Fs[:args.num_samples]
    results = Parallel(
        n_jobs=n_jobs,
        backend="multiprocessing",
        verbose=True
    )(
        delayed(_run)(F, ring, k_last_calls=k_last_calls)
        for F in Fs
    )
    samples = list(it.chain(*results))
    Ls, Vs = zip(*samples)
    return Ls, Vs
    

def output_csv(overall_stats, args):
    # Make it into csv and save
    rows = []
    for (p, n, k), stats in overall_stats.items():
        row = {
            "p": p,
            "n": n,
            "k": k,
        }
        # Flatten various statistics
        for key, val in stats.items():
            for stat_name, stat_val in val.items():
                row[f"{key}_{stat_name}"] = stat_val
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"results/eval/infix_vs_monomial_token_counts_num_samples={args.num_samples}.csv", index=False)
    return df 

def plot_fig_V(overall_stats, args):
    p_list = [7, 31, 127]
    n_list = [3, 4, 5]
    k_list = [1, 3, 5]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
    bar_width = 0.25
    x = np.arange(len(n_list))
    colors = plt.get_cmap("Set2").colors

    for i, p in enumerate(p_list):
        for j, k in enumerate(k_list):
            ax = axs[i, j]
            full_infix_tokens = []
            scaled_infix_tokens = []
            monomial_tokens = []
            for n in n_list:
                key_all = (p, n, None)
                key_k = (p, n, k)
                if key_all in overall_stats:
                    full_infix_tokens.append(overall_stats[key_all]["infix_token_counts_V"]["max"])
                else:
                    full_infix_tokens.append(0)
                if key_k in overall_stats:
                    scaled_infix_tokens.append(overall_stats[key_k]["infix_token_counts_V"]["max"])
                    monomial_tokens.append(overall_stats[key_k]["monomial_token_counts_V"]["max"])
                else:
                    scaled_infix_tokens.append(0)
                    monomial_tokens.append(0)
            print(f"p={p}, k={k}, full_infix={full_infix_tokens}, scaled_infix={scaled_infix_tokens}, monomial={monomial_tokens}")
            bars1 = ax.bar(x - bar_width, full_infix_tokens, width=bar_width, label=f"All Terms (Infix)", color=colors[0])
            bars2 = ax.bar(x, scaled_infix_tokens, width=bar_width, label=f"$l$ Terms (Infix)", color=colors[1])
            bars3 = ax.bar(x + bar_width, monomial_tokens, width=bar_width, label=f"$l$ Terms (Monomial)", color=colors[2])

            max_val = max([*full_infix_tokens, *scaled_infix_tokens, *monomial_tokens])
            if max_val > 0:
                ax.set_ylim(top=max_val * 2.5)
            else:
                ax.set_ylim(top=1)

            ax.bar_label(bars1, labels=[f"{int(v)}" if v > 0 else "" for v in full_infix_tokens], fontsize=14, padding=0)
            ax.bar_label(bars2, labels=[f"{int(v)}" if v > 0 else "" for v in scaled_infix_tokens], fontsize=14, padding=0)
            ax.bar_label(bars3, labels=[f"{int(v)}" if v > 0 else "" for v in monomial_tokens], fontsize=14, padding=0)

            ax.set_title(f"$\\mathbb{{F}}_{{{p}}}$, $l={k}$", fontsize=24)
            ax.set_xticks(x)
            ax.set_xticklabels([f"$n={n}$" for n in n_list], fontsize=16)
            ax.set_xlabel("Number of Variables (n)", fontsize=20)
            if j == 0:
                ax.set_ylabel("Max Token Count", fontsize=20)
            ax.set_yscale('log')
            ax.set_axisbelow(True)
            ax.grid(True, axis='y')
            ax.tick_params(axis='y', labelsize=16)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(1.00, 1.000),
        fontsize=20,
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/eval/V_token_count_num_samples={args.num_samples}.pdf")

def plot_fig_LV(overall_stats, args):
    p_list = [7, 31, 127]
    n_list = [3, 4, 5]
    k_list = [1, 3, 5]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
    bar_width = 0.25
    x = np.arange(len(n_list))

    pastel = plt.get_cmap("Set2").colors

    base_colors = [pastel[2], pastel[0], pastel[1]]  # pink, light green, light blue
    # Create light colors with alpha (L: base, V: alpha=0.5)
    def with_alpha(rgb, alpha):
        return (rgb[0], rgb[1], rgb[2], alpha)
    light_colors = [
        with_alpha(pastel[2], 0.5),
        with_alpha(pastel[0], 0.5),
        with_alpha(pastel[1], 0.5),
    ]

    for i, p in enumerate(p_list):
        for j, k in enumerate(k_list):
            ax = axs[i, j]
            # All Terms (Infix)
            infix_all_L = []
            infix_all_V = []
            # l Terms (Infix)
            infix_k_L = []
            infix_k_V = []
            # l Terms (Monomial)
            monomial_L = []
            monomial_V = []
            for n in n_list:
                key_all = (p, n, None)
                key_k = (p, n, k)
                # All Terms (Infix)
                if key_all in overall_stats:
                    infix_all_L.append(overall_stats[key_all]["infix_token_counts_L"]["max"])
                    infix_all_V.append(overall_stats[key_all]["infix_token_counts_V"]["max"])
                else:
                    infix_all_L.append(0)
                    infix_all_V.append(0)
                # l Terms (Infix) & Monomial
                if key_k in overall_stats:
                    infix_k_L.append(overall_stats[key_k]["infix_token_counts_L"]["max"])
                    infix_k_V.append(overall_stats[key_k]["infix_token_counts_V"]["max"])
                    monomial_L.append(overall_stats[key_k]["monomial_token_counts_L"]["max"])
                    monomial_V.append(overall_stats[key_k]["monomial_token_counts_V"]["max"])
                else:
                    infix_k_L.append(0)
                    infix_k_V.append(0)
                    monomial_L.append(0)
                    monomial_V.append(0)
            # Stacked bar chart
            bars1 = ax.bar(x - bar_width, infix_all_L, width=bar_width, label="All Terms $L$ (Infix)", color=base_colors[0])
            bars1b = ax.bar(x - bar_width, infix_all_V, width=bar_width, bottom=infix_all_L, label="All Terms $V$ (Infix)", color=light_colors[0])
            bars2 = ax.bar(x, infix_k_L, width=bar_width, label=f"$l$ Terms $L$ (Infix)", color=base_colors[1])
            bars2b = ax.bar(x, infix_k_V, width=bar_width, bottom=infix_k_L, label=f"$l$ Terms $V$ (Infix)", color=light_colors[1])
            bars3 = ax.bar(x + bar_width, monomial_L, width=bar_width, label=f"$l$ Terms $L$ (Monomial)", color=base_colors[2])
            bars3b = ax.bar(x + bar_width, monomial_V, width=bar_width, bottom=monomial_L, label=f"$l$ Terms $V$ (Monomial)", color=light_colors[2])

            max_val = max([
                *(np.array(infix_all_L) + np.array(infix_all_V)),
                *(np.array(infix_k_L) + np.array(infix_k_V)),
                *(np.array(monomial_L) + np.array(monomial_V))
            ])
            if max_val > 0:
                ax.set_ylim(top=max_val * 5)
            else:
                ax.set_ylim(top=1)

            ax.bar_label(bars1, labels=[f"{int(v)}" if v > 0 else "" for v in infix_all_L], fontsize=14, padding=0)
            ax.bar_label(bars1b, labels=[f"{int(v)}" if v > 0 else "" for v in infix_all_V], fontsize=14, padding=0)
            ax.bar_label(bars2, labels=[f"{int(v)}" if v > 0 else "" for v in infix_k_L], fontsize=14, padding=0)
            ax.bar_label(bars2b, labels=[f"{int(v)}" if v > 0 else "" for v in infix_k_V], fontsize=14, padding=0)
            ax.bar_label(bars3, labels=[f"{int(v)}" if v > 0 else "" for v in monomial_L], fontsize=14, padding=0)
            ax.bar_label(bars3b, labels=[f"{int(v)}" if v > 0 else "" for v in monomial_V], fontsize=14, padding=0)

            ax.set_title(f"$\\mathbb{{F}}_{{{p}}}$, $l={k}$", fontsize=24)
            ax.set_xticks(x)
            ax.set_xticklabels([f"$n={n}$" for n in n_list], fontsize=16)
            ax.set_xlabel("Number of Variables (n)", fontsize=20)
            if j == 0:
                ax.set_ylabel("Max Token Count", fontsize=20)
            ax.set_yscale('log')
            ax.set_axisbelow(True)
            ax.grid(True, axis='y')
            ax.tick_params(axis='y', labelsize=16)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(1.00, 1.000),
        fontsize=20,
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(f"results/eval/LV_token_count_max_num_samples={args.num_samples}.pdf")

def plot_fig_LV_sum(df, args):
    p_list = [7, 31, 127]
    n_list = [3, 4, 5]
    k_list = [1, 3, 5]
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
    bar_width = 0.25
    x = np.arange(len(n_list))
    colors = plt.get_cmap("Set2").colors

    for i, p in enumerate(p_list):
        for j, k in enumerate(k_list):
            ax = axs[i, j]
            all_terms = []
            k_terms = []
            monomial_terms = []
            for n in n_list:
                row_all = df[(df['p'] == p) & (df['n'] == n) & (df['k'].isnull())]
                row_k = df[(df['p'] == p) & (df['n'] == n) & (df['k'] == k)]
                if not row_all.empty and not row_k.empty:
                    all_terms.append(row_all.iloc[0]['infix_token_counts_L_max'] + row_all.iloc[0]['infix_token_counts_V_max'])
                    k_terms.append(row_k.iloc[0]['infix_token_counts_L_max'] + row_k.iloc[0]['infix_token_counts_V_max'])
                    monomial_terms.append(row_k.iloc[0]['monomial_token_counts_L_max'] + row_k.iloc[0]['monomial_token_counts_V_max'])
                else:
                    all_terms.append(np.nan)
                    k_terms.append(np.nan)
                    monomial_terms.append(np.nan)
            bars1 = ax.bar(x - bar_width, all_terms, width=bar_width, label="All Terms (Infix)", color=colors[0])
            bars2 = ax.bar(x, k_terms, width=bar_width, label=f"$l$ Terms (Infix)", color=colors[1])
            bars3 = ax.bar(x + bar_width, monomial_terms, width=bar_width, label=f"$l$ Terms (Monomial)", color=colors[2])
            max_val = np.nanmax([*all_terms, *k_terms, *monomial_terms])
            if max_val > 0:
                ax.set_ylim(top=max_val * 2.5)
            ax.bar_label(bars1, labels=[f"{v:.0f}" if not np.isnan(v) and v > 0 else "" for v in all_terms], fontsize=14, padding=0)
            ax.bar_label(bars2, labels=[f"{v:.0f}" if not np.isnan(v) and v > 0 else "" for v in k_terms], fontsize=14, padding=0)
            ax.bar_label(bars3, labels=[f"{v:.0f}" if not np.isnan(v) and v > 0 else "" for v in monomial_terms], fontsize=14, padding=0)
            ax.set_title(f"$\\mathbb{{F}}_{{{p}}}$, $l={k}$", fontsize=24)
            ax.set_xticks(x)
            ax.set_xticklabels([f"$n={n}$" for n in n_list], fontsize=16)
            ax.set_xlabel("Number of Variables (n)", fontsize=20)
            if j == 0:
                ax.set_ylabel("Max Token Count", fontsize=20)
            ax.set_yscale('log')
            ax.set_axisbelow(True)
            ax.grid(True, axis='y')
            ax.tick_params(axis='y', labelsize=16)
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(1.00, 1.000),
        fontsize=20,
        ncol=3,
        frameon=True
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/eval/LV_token_count_max_num_samples={args.num_samples}.pdf")

def plot_fig_LV_mean(overall_stats, args):
    p_list = [7, 31, 127]
    n_list = [3, 4, 5]
    k_list = [1, 3, 5]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
    bar_width = 0.25
    x = np.arange(len(n_list))

    pastel = plt.get_cmap("Set2").colors
    base_colors = [pastel[2], pastel[0], pastel[1]]
    def with_alpha(rgb, alpha):
        return (rgb[0], rgb[1], rgb[2], alpha)
    light_colors = [
        with_alpha(pastel[2], 0.5),
        with_alpha(pastel[0], 0.5),
        with_alpha(pastel[1], 0.5),
    ]

    for i, p in enumerate(p_list):
        for j, k in enumerate(k_list):
            ax = axs[i, j]
            infix_all_L = []
            infix_all_V = []
            infix_k_L = []
            infix_k_V = []
            monomial_L = []
            monomial_V = []
            for n in n_list:
                key_all = (p, n, None)
                key_k = (p, n, k)
                if key_all in overall_stats:
                    infix_all_L.append(overall_stats[key_all]["infix_token_counts_L"]["mean"])
                    infix_all_V.append(overall_stats[key_all]["infix_token_counts_V"]["mean"])
                else:
                    infix_all_L.append(0)
                    infix_all_V.append(0)
                if key_k in overall_stats:
                    infix_k_L.append(overall_stats[key_k]["infix_token_counts_L"]["mean"])
                    infix_k_V.append(overall_stats[key_k]["infix_token_counts_V"]["mean"])
                    monomial_L.append(overall_stats[key_k]["monomial_token_counts_L"]["mean"])
                    monomial_V.append(overall_stats[key_k]["monomial_token_counts_V"]["mean"])
                else:
                    infix_k_L.append(0)
                    infix_k_V.append(0)
                    monomial_L.append(0)
                    monomial_V.append(0)
            bars1 = ax.bar(x - bar_width, infix_all_L, width=bar_width, label="All Terms L (Infix)", color=base_colors[0])
            bars1b = ax.bar(x - bar_width, infix_all_V, width=bar_width, bottom=infix_all_L, label="All Terms V (Infix)", color=light_colors[0])
            bars2 = ax.bar(x, infix_k_L, width=bar_width, label=f"$l Terms L (Infix)$", color=base_colors[1])
            bars2b = ax.bar(x, infix_k_V, width=bar_width, bottom=infix_k_L, label=f"$l Terms V (Infix)$", color=light_colors[1])
            bars3 = ax.bar(x + bar_width, monomial_L, width=bar_width, label=f"$l Terms L (Monomial)$", color=base_colors[2])
            bars3b = ax.bar(x + bar_width, monomial_V, width=bar_width, bottom=monomial_L, label=f"$l Terms V (Monomial)$", color=light_colors[2])

            max_val = max([
                *(np.array(infix_all_L) + np.array(infix_all_V)),
                *(np.array(infix_k_L) + np.array(infix_k_V)),
                *(np.array(monomial_L) + np.array(monomial_V))
            ])
            if max_val > 0:
                ax.set_ylim(top=max_val * 5)
            else:
                ax.set_ylim(top=1)

            ax.bar_label(bars1, labels=[f"{v:.1f}" if v > 0 else "" for v in infix_all_L], fontsize=14, padding=0)
            ax.bar_label(bars1b, labels=[f"{v:.1f}" if v > 0 else "" for v in infix_all_V], fontsize=14, padding=0)
            ax.bar_label(bars2, labels=[f"{v:.1f}" if v > 0 else "" for v in infix_k_L], fontsize=14, padding=0)
            ax.bar_label(bars2b, labels=[f"{v:.1f}" if v > 0 else "" for v in infix_k_V], fontsize=14, padding=0)
            ax.bar_label(bars3, labels=[f"{v:.1f}" if v > 0 else "" for v in monomial_L], fontsize=14, padding=0)
            ax.bar_label(bars3b, labels=[f"{v:.1f}" if v > 0 else "" for v in monomial_V], fontsize=14, padding=0)

            ax.set_title(f"$\\mathbb{{F}}_{{{p}}}$, $l={k}$", fontsize=24)
            ax.set_xticks(x)
            ax.set_xticklabels([f"$n={n}$" for n in n_list], fontsize=16)
            ax.set_xlabel("Number of Variables (n)", fontsize=20)
            if j == 0:
                ax.set_ylabel("Mean Token Count", fontsize=20)
            ax.set_yscale('log')
            ax.set_axisbelow(True)
            ax.grid(True, axis='y')
            ax.tick_params(axis='y', labelsize=16)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(1.00, 1.000),
        fontsize=20,
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(f"results/eval/LV_token_count_mean_num_samples={args.num_samples}.pdf")

def load_overall_stats_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    overall_stats = {}
    for _, row in df.iterrows():
        p = int(row['p'])
        n = int(row['n'])
        k = row['k']
        if pd.isnull(k):
            k = None
        else:
            k = int(k)
        stats = {}
        for col in row.index:
            if col in ['p', 'n', 'k']:
                continue
            # e.g. infix_token_counts_L_mean
            if '_' in col:
                key, stat = col.rsplit('_', 1)
                if key not in stats:
                    stats[key] = {}
                stats[key][stat] = row[col]
        overall_stats[(p, n, k)] = stats
    return overall_stats

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.from_csv:
        csv_path = f"results/eval/infix_vs_monomial_token_counts_num_samples={args.num_samples}.csv"
        overall_stats = load_overall_stats_from_csv(csv_path)
        plot_fig_V(overall_stats, args)
        plot_fig_LV(overall_stats, args)
        plot_fig_LV_mean(overall_stats, args)
        return

    overall_stats = {}  # initialize

    for p, n in tqdm(list(it.product([7, 31, 127], [3, 4, 5]))):
        config = f"config/problems/border_basis_GF{p}_n={n}.yaml"
        ring = PolynomialRing(GF(p), n, 'x', order='degrevlex')
        degree_bounds = '_'.join([str(4)] * n)
        dirname = Path(f'GF{p}_n={n}_deg=1_terms=10_bounds={degree_bounds}_total={2}')
        source_data_path = f"./data/border_basis/{dirname}"
        Ls, Vs = run(args, config, source_data_path, n_jobs=args.n_jobs, k_last_calls=args.k_last_calls)

        for k in [None, 1, 3, 5]:
            polyseq_tosequence = lambda polyseq: ' [SEP] '.join([poly_to_sequence(p) for p in polyseq])
            leading_terms_processor = get_leading_terms_processor(k)
            LVs_text = [f'{polyseq_tosequence(L)} [BIGSEP] {polyseq_tosequence(V)}' for L, V in zip(Ls, Vs)]
            
            if k: LVs_text = leading_terms_processor.process(LVs_text)
            
            Ls_text = [LV.split(' [BIGSEP] ')[0] for LV in LVs_text]
            Vs_text = [LV.split(' [BIGSEP] ')[1] for LV in LVs_text]
            
            monomial_processor = get_monomial_processor(ring)
            Vs_monomial = monomial_processor(Vs_text)
            
            infix_token_counts_L = [len(text.split()) for text in Ls_text]
            infix_token_counts_V = [len(text.split()) for text in Vs_text]
            infix_token_counts_LV = [len(text.split()) for text in LVs_text]
            
            monomial_token_counts_L = [len(L) for L in Ls]
            monomial_token_counts_V = [len(V) - 1 for V in Vs_monomial] # -1 to remove bos
            monomial_token_counts_LV = [mtc_L + mtc_V for mtc_L, mtc_V in zip(monomial_token_counts_L, monomial_token_counts_V)]
            
            # summarize the statistics (mean, min, max, std) of each count
            stats = {
                "infix_token_counts_L": {
                    "mean": np.mean(infix_token_counts_L),
                    "min": np.min(infix_token_counts_L),
                    "max": np.max(infix_token_counts_L),
                    "std": np.std(infix_token_counts_L)
                },
                "infix_token_counts_V": {
                    "mean": np.mean(infix_token_counts_V),
                    "min": np.min(infix_token_counts_V),
                    "max": np.max(infix_token_counts_V),
                    "std": np.std(infix_token_counts_V)
                },
                "infix_token_counts_LV": {
                    "mean": np.mean(infix_token_counts_LV),
                    "min": np.min(infix_token_counts_LV),
                    "max": np.max(infix_token_counts_LV),
                    "std": np.std(infix_token_counts_LV)
                },
                "monomial_token_counts_L": {
                    "mean": np.mean(monomial_token_counts_L),
                    "min": np.min(monomial_token_counts_L),
                    "max": np.max(monomial_token_counts_L),
                    "std": np.std(monomial_token_counts_L)
                },
                "monomial_token_counts_V": {
                    "mean": np.mean(monomial_token_counts_V),
                    "min": np.min(monomial_token_counts_V),
                    "max": np.max(monomial_token_counts_V),
                    "std": np.std(monomial_token_counts_V)
                },
                "monomial_token_counts_LV": {
                    "mean": np.mean(monomial_token_counts_LV),
                    "min": np.min(monomial_token_counts_LV),
                    "max": np.max(monomial_token_counts_LV),
                    "std": np.std(monomial_token_counts_LV)
                }
            }
            overall_stats[(p, n, k)] = stats  # add to the overall dictionary
    
    df = output_csv(overall_stats, args)
    plot_fig_V(overall_stats, args)
    plot_fig_LV(overall_stats, args)
    plot_fig_LV_mean(overall_stats, args)
    # plot_fig_LV_sum(df, args)


if __name__ == "__main__":
    main()