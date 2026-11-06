import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 evaluation for multiple models and jobs.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run by passing --resume to run_pipeline.py")
    args = parser.parse_args()

    occupations = [
        "Accountants", "Web Developers", "Computer Programmers", 
        "Financial Quantitative", "Market Research", "Lawyers", 
        "Technical Writers", "Statisticians", "Economists", "Logistics Analysts"
    ]

    models = [
        {"name": "qwen2.5:1.5b", "dir": "results_1.5b_v2", "desc": "1.5b model with 6 dimensions"},
        {"name": "qwen2.5:7b", "dir": "results_7b", "desc": "7b model with 6 dimensions"}
    ]

    for model in models:
        print(f"\n========================================================")
        print(f"Running {model['desc']} ({model['dir']}/)")
        print(f"========================================================\n")
        
        for job in occupations:
            print(f"--- Running {job} for {model['name']} ---")
            
            cmd = [
                "python", "run_pipeline.py",
                "--job", job,
                "--student-model", model['name'],
                "--results-dir", model['dir']
            ]
            
            if args.resume:
                cmd.append("--resume")
                
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running pipeline for {job} on model {model['name']}. Exit code: {e.returncode}")
                # Continue to the next job instead of exiting entirely
                continue

if __name__ == "__main__":
    main()
