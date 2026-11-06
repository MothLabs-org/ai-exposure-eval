import os
import subprocess
import argparse

def pull_model(model_name):
    print(f"Ensuring model {model_name} is available locally...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to pull {model_name}. It might not be available or Ollama is not running. Exit code: {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 evaluation for multiple new student models.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run by passing --resume to run_pipeline.py")
    args = parser.parse_args()

    occupations = [
        "Accountants", "Web Developers", "Computer Programmers", 
        "Financial Quantitative", "Market Research", "Lawyers", 
        "Technical Writers", "Statisticians", "Economists", "Logistics Analysts"
    ]

    models = [
        {"name": "llama3:8b", "dir": "results_llama3", "desc": "Meta Llama 3 (8B)"},
        {"name": "mistral:v0.3", "dir": "results_mistral", "desc": "Mistral v0.3"},
        {"name": "phi3:mini", "dir": "results_phi3", "desc": "Microsoft Phi-3 Mini"},
        {"name": "gemma2:9b", "dir": "results_gemma2", "desc": "Google Gemma 2 (9B)"}
    ]

    # Pre-pull all models to prevent download timeouts during evaluation
    for model in models:
        pull_model(model['name'])

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
