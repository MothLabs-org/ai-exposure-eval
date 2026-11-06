#!/usr/bin/env bash

OCCUPATIONS=("Accountants" "Software Dev" "Computer Programmer" "Financial Analyst" "Market Research" "Lawyer" "Technical Writer" "Statistician" "Web Dev" "Economist")

MODELS=("qwen2.5:1.5b" "qwen2.5:7b")
DIRS=("results_1.5b_v2" "results_7b")
DESCS=("1.5b model with 6 dimensions" "7b model with 6 dimensions")

RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "Resume mode enabled. Will skip already completed tasks."
fi

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    dir="${DIRS[$i]}"
    desc="${DESCS[$i]}"
    
    echo ""
    echo "========================================================"
    echo "Running ${desc} (${dir}/)"
    echo "========================================================"
    echo ""
    
    for job in "${OCCUPATIONS[@]}"; do
        echo "--- Running $job for $model ---"
        python run_pipeline.py \
            --job "$job" \
            --student-model "$model" \
            --results-dir "$dir" \
            $RESUME_FLAG || echo "Error running $job for $model. Continuing to next..."
    done
done
