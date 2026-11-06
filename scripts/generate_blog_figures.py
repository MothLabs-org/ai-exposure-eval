import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure matplotlib for a modern, clean look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'sans-serif']

def load_results(base_dir="."):
    """Load evaluation results from various model runs."""
    results = []
    
    # Define the directories and their associated model sizes
    runs = [
        {"dir": "results_1.5b_v2", "model": "Qwen 1.5B"},
        {"dir": "results_gemma3_4b", "model": "Gemma 3 4B"},
        {"dir": "results", "model": "Qwen 35B"}
    ]
    
    for run in runs:
        path = os.path.join(base_dir, run["dir"], "*.json")
        for file in glob.glob(path):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    meta = data.get("meta", {})
                    job_title = meta.get("job_title", "Unknown")
                    job_id = meta.get("job_id", "Unknown")
                    exposure = meta.get("exposure_score", {})
                    overall = exposure.get("overall", 0)
                    dims = exposure.get("dimensions", {})
                    
                    results.append({
                        "Job Title": job_title,
                        "Job ID": job_id,
                        "Model": run["model"],
                        "Overall Score": overall,
                        "Correctness": dims.get("Correctness", 0),
                        "Completeness": dims.get("Completeness", 0),
                        "Best Practices": dims.get("Best Practices", 0),
                        "Domain Accuracy": dims.get("Domain Accuracy", 0),
                        "Clarity": dims.get("Clarity", 0),
                        "Autonomy": dims.get("Autonomy", 0)
                    })
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    return pd.DataFrame(results)

def plot_scaling_heatmap(df, output_path="scaling_heatmap.png"):
    """Plot a heatmap showing how overall exposure scales with model size."""
    # Filter to jobs that have runs across multiple models (or a selected subset)
    # The 7B and 4B models currently have few jobs run (e.g. Accountants, Computer Programmers, Web Devs)
    subset_jobs = ["Accountants and Auditors", "Computer Programmers", "Web Developers"]
    df_sub = df[df["Job Title"].isin(subset_jobs)].copy()
    
    if df_sub.empty:
        print("Not enough overlapping data for scaling heatmap.")
        return
        
    pivot = df_sub.pivot(index="Job Title", columns="Model", values="Overall Score")
    
    cols = [c for c in ["Qwen 1.5B", "Gemma 3 4B", "Qwen 35B"] if c in pivot.columns]
    pivot = pivot[cols]
    
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=1, vmax=5, fmt=".2f", linewidths=.5)
    plt.title("AI Exposure Score Scaling by Model Size", pad=20, fontweight="bold")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_fluent_but_flawed(df, output_path="fluent_but_flawed.png"):
    """Plot the gap between Clarity and Domain Accuracy to illustrate Sycophancy."""
    # Focus on the 1.5B model for the core finding
    df_15b = df[df["Model"] == "Qwen 1.5B"].copy()
    
    if df_15b.empty:
        print("No 1.5B data found for fluent-but-flawed chart.")
        return
        
    # Select a few subjective/advisory vs objective roles
    jobs_of_interest = [
        "Accountants and Auditors", 
        "Lawyers", 
        "Market Research Analysts",
        "Computer Programmers", 
        "Software Developers"
    ]
    
    df_15b['Gap (Clarity - Domain)'] = df_15b['Clarity'] - df_15b['Domain Accuracy']
    
    available_jobs = df_15b[df_15b["Job Title"].isin(jobs_of_interest)]
    if available_jobs.empty:
        available_jobs = df_15b.sort_values(by="Gap (Clarity - Domain)", ascending=False).head(6)
        
    available_jobs = available_jobs.sort_values(by="Gap (Clarity - Domain)", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    y = np.arange(len(available_jobs))
    height = 0.35
    
    ax.barh(y - height/2, available_jobs["Domain Accuracy"], height, label='Domain Accuracy', color='#e74c3c')
    ax.barh(y + height/2, available_jobs["Clarity"], height, label='Clarity', color='#3498db')
    
    ax.set_yticks(y)
    ax.set_yticklabels(available_jobs["Job Title"])
    #ax.set_xlim(0, 5)
    ax.set_xlabel('Score (1-5)')
    ax.set_title('The "Fluent but Flawed" Gap (Qwen 1.5B)', pad=20, fontweight="bold")
    ax.legend(loc='lower right')
    
    # Add gap text
    for i, (_, row) in enumerate(available_jobs.iterrows()):
        gap = row["Gap (Clarity - Domain)"]
        ax.text(max(row["Clarity"], row["Domain Accuracy"]) + 0.1, i, f"Gap: {gap:.2f}", va='center', fontstyle='italic', color='#555555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_dimension_radar(df, output_path="dimension_radar.png"):
    """Plot a radar chart for a specific job across models (5 shared dimensions)."""
    job_target = "Accountants and Auditors"
    df_job = df[df["Job Title"] == job_target]
    
    if df_job.empty:
        print(f"No data for {job_target} to plot radar chart.")
        return

    # Use only the 5 dimensions shared across all models
    categories = ['Correctness', 'Completeness', 'Best Practices', 'Domain Accuracy', 'Clarity']
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
    plt.ylim(0, 5)

    colors = ['#3498db', '#e67e22', '#2ecc71']
    for i, (_, row) in enumerate(df_job.iterrows()):
        values = row[categories].tolist()
        values += values[:1]
        model_name = row["Model"]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i%len(colors)])
        ax.fill(angles, values, color=colors[i%len(colors)], alpha=0.1)

    plt.title(f"Dimension Profile: {job_target}", size=15, fontweight="bold", pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_autonomy_comparison(df, output_path="autonomy_comparison.png"):
    """Plot Autonomy scores specifically for models that have the 6th dimension."""
    # Filter for models that actually have non-zero autonomy
    df_auto = df[df['Autonomy'] > 0].copy()
    
    if df_auto.empty:
        print("No autonomy data found.")
        return
        
    jobs_of_interest = ["Accountants and Auditors", "Computer Programmers", "Web Developers"]
    df_auto = df_auto[df_auto["Job Title"].isin(jobs_of_interest)]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_auto, x="Job Title", y="Autonomy", hue="Model", palette="viridis")
    
    plt.ylim(0, 5)
    plt.title("Autonomy Scores Across Occupations", pad=20, fontweight="bold")
    plt.ylabel("Score (1-5)")
    plt.xlabel("")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".", help="Base directory of the eval repository")
    parser.add_argument("--out-dir", default="data/figures", help="Output directory for figures")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df = load_results(args.base_dir)
    print(f"Loaded {len(df)} job evaluations.")
    
    if not df.empty:
        plot_scaling_heatmap(df, os.path.join(args.out_dir, "scaling_heatmap.png"))
        plot_fluent_but_flawed(df, os.path.join(args.out_dir, "fluent_but_flawed.png"))
        plot_dimension_radar(df, os.path.join(args.out_dir, "dimension_radar.png"))
        plot_autonomy_comparison(df, os.path.join(args.out_dir, "autonomy_comparison.png"))
        
        # Save aggregated CSV for the blog post
        df.to_csv(os.path.join(args.out_dir, "aggregated_results.csv"), index=False)
        print(f"Saved aggregated data to {os.path.join(args.out_dir, 'aggregated_results.csv')}")
    else:
        print("No data found to plot.")
