import json
import glob
import os
import pandas as pd

def analyze_results(directory, model_name):
    files = glob.glob(os.path.join(directory, "*.json"))
    records = []
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as infile:
            try:
                data = json.load(infile)
                meta = data.get("meta", {})
                job = meta.get("job_title", "Unknown")
                soc = meta.get("onet_code", "Unknown")
                scores = meta.get("exposure_score", {})
                
                record = {
                    "Model": model_name,
                    "Job": job,
                    "SOC": soc,
                    "Overall": scores.get("overall", 0),
                }
                # Add individual dimension scores
                for dim, score in scores.get("dimensions", {}).items():
                    record[dim] = score
                    
                records.append(record)
            except Exception as e:
                print(f"Error parsing {f}: {e}")
                
    return pd.DataFrame(records)

def main():
    dfs = []
    
    # Dynamically find all results directories
    for d in glob.glob("results*"):
        if os.path.isdir(d):
            model_name = d.replace("results_", "")
            if model_name == "results": model_name = "default"
            df = analyze_results(d, model_name)
            if not df.empty:
                dfs.append(df)
                
    if not dfs:
        print("No results found yet.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
        
    print("\n--- Dimension Decomposition & Sycophancy Analysis ---")
    
    # Calculate the gap between Clarity and Domain Accuracy
    if "Clarity" in df_all.columns and "Domain Accuracy" in df_all.columns:
        df_all["Fluent_Flawed_Gap"] = df_all["Clarity"] - df_all["Domain Accuracy"]
        
        # Sort by gap to see which jobs have the biggest 'Fluent but Flawed' effect
        df_sorted = df_all.sort_values("Fluent_Flawed_Gap", ascending=False)
        print("\nTop 10 Most Sycophantic Evaluations ('Fluent but Flawed' Gap):")
        print(df_sorted[["Model", "Job", "Clarity", "Domain Accuracy", "Fluent_Flawed_Gap"]].head(10))
        
        print("\nAverage Sycophancy Gap by Model:")
        avg_gap = df_all.groupby("Model")["Fluent_Flawed_Gap"].mean().sort_values(ascending=False)
        print(avg_gap)
    
    print("\nDetailed summary saved to dimension_analysis_summary.csv")
    df_all.to_csv("dimension_analysis_summary.csv", index=False)

if __name__ == "__main__":
    main()
