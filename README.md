# AI Occupational Exposure Evaluation Framework

> Extending AI exposure scoring across 114 occupations — with more student models,
> an Autonomy dimension and a fluent-but-flawed analysis.

**Author:** Hao Zhang · UCLA M.Eng.  
**Original Work:** Yiru Chen  
**Mentor:** Xibin Gao · [Moth Labs](https://www.mothlabs.org/)

The results are written up in the Moth Labs post [Measuring What AI Can Actually Do: Occupational Exposure Beyond Fluent-but-Flawed Outputs](https://www.mothlabs.org/blog/the-fluent-but-flawed-gap-measuring-ai-occupational-exposure-beyond-synthetic-feasibility). You can explore them in the [live dashboard](https://www.mothlabs.org/ai-exposure/).

---

## Background

This project builds on the `job_eval` pipeline created by Yiru Chen during her internship at Moth Labs; the `results/` run comes from that original pipeline. The original system quantifies AI exposure risk across 114 real-world occupations using O*NET® task data via a 3-step LLM evaluation pipeline (Teacher → Student → Judge).

This repository aims to extend that work into a rigorous, reproducible, and publishable research contribution aligned with Moth Labs' mission of AI safety, transparency, and accessibility.

## Research Goals (Hao's Research)

- **Task-Level Decomposition**: Extending scoring from occupation-level to O*NET Detailed Work Activity (DWA) level.
- **Multi-Model Evaluation**: Running exposure assessments using ≥3 LLMs (e.g., GPT-4o, Claude 3.5, Qwen3.5, Llama-3-70B) to measure cross-model variance.
- **Agentic Exposure Dimension**: Designing rubrics for tool-using AI agent evaluation (beyond text-in/text-out).
- **Empirical Validation**: Quantifying the gap between theoretical exposure and empirical completion quality via controlled experiments.

Done in this repository: the Autonomy dimension, the Gemma 3 4B student run, a partial Qwen2.5 1.5B student run (3 occupations), and the gap between Clarity and Domain Accuracy scores (`analyze_dimensions.py`). Not yet in this repository: scoring at the O\*NET Detailed Work Activity (DWA) level, runs with hosted models such as GPT-4o or Claude, rubrics for tool-using agents, and a comparison of these scores with theoretical exposure estimates.

---

## Overview

Quantify **AI exposure risk** across 114 real-world occupations using a three-step LLM evaluation pipeline grounded in [O\*NET](https://www.onetonline.org/) task data.

Works with **any OpenAI-compatible model server** — vLLM, Ollama, OpenAI API, etc.

---

## Architecture

```
Task Statements.xlsx  (O*NET task data)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  For each occupation (114 total)                    │
│                                                     │
│  ① Config    LLM generates job evaluation config    │
│      │        (cached in configs/ after first run)  │
│      ▼                                              │
│  ② Step 1   Teacher LLM → evaluation tasks         │
│      ↓        (5 tasks per batch from O*NET data)   │
│  ③ Step 2   Student LLM → answers each task        │
│      ↓                                              │
│  ④ Step 3   Judge LLM  → scores 6 dimensions       │
│                           (1–5 per task)            │
└─────────────────────────────────────────────────────┘
         │
         ▼
results/{job_id}_results_auto.json
         │
         ▼
streamlit run dashboard.py   →   interactive analytics
```

**Six scoring dimensions** (applied uniformly across all occupations):

| # | Dimension | What it measures |
|---|---|---|
| 1 | Correctness | Does the answer solve the task? |
| 2 | Completeness | Are all parts fully addressed? |
| 3 | Best Practices | Does it follow professional standards? |
| 4 | Domain Accuracy | Are domain-specific facts and terms correct? |
| 5 | Clarity | Is the output well-structured and readable? |
| 6 | Autonomy | Could the output be used as-is without human review (5), or does it need significant rework (1)? |

The first run (`results/`) predates Autonomy and is scored on the first five.

The **Exposure Score** (1–5) is the mean across all dimensions and all graded tasks for a given occupation. Higher = the judge rated the student model's answers to that occupation's generated tasks more highly; it is not a direct measure of how much of the real job a model can do.

---

## Results in This Repository

| Directory | Student model | Occupations | Tasks | Dimensions |
|---|---|---|---|---|
| `results/` | Qwen/Qwen3.5-35B-A3B-FP8 | 114 | 2,430 | 5 |
| `results_gemma3_4b/` | gemma-3-4b | 114 | 2,340 | 6 |
| `results_1.5b_v2/` | qwen2.5:1.5b (partial run) | 3 | 75 | 6 |

In every run, Qwen/Qwen3.5-35B-A3B-FP8 wrote the tasks (teacher) and graded the answers (judge). In `results/` it also wrote the answers, so that run grades its own work. The result files have no `judge_model` field, because the code that produced them always used the teacher as judge. Each run generated its own tasks from largely the same O\*NET task statements, so the runs share statements but few prompts. The Tasks column counts every task, including 72, 30 and 12 tasks, respectively, that got no usable grade from the judge (`score` 0 and empty `dimension_scores`); the exposure scores leave those out, though `evaluated_tasks` still counts them. Overall scores average five dimensions in `results/` and six in the other runs.

You don't need a model server to explore the results: open the [live dashboard](https://www.mothlabs.org/ai-exposure/), or install the requirements (step 1 of the Quick Start) and run `streamlit run dashboard.py`, which reads the committed result files.

The blog post's figures are in `data/figures/`. `scripts/generate_blog_figures.py` regenerates them; it also needs `seaborn` (`pip install seaborn`), which `requirements.txt` doesn't include.

**The tasks, answers and grades in the result files are unedited model output**, except that 106 tasks were removed from 42 `results/` files because the file already had a task for the same O\*NET statement (the first one was kept), and the scores in 49 `results/` files were recalculated. The answers include invented people, email addresses, citations and facts, some attached to real people, events and organizations. Don't treat them as real.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/MothLabs-org/ai-exposure-eval.git
cd ai-exposure-eval
pip install -r requirements.txt -c constraints.txt

# 2. Configure your model server
cp pipeline_config.example.json pipeline_config.json
# Edit pipeline_config.json — set "server.ip" and "models.teacher" / "models.student"

# 3. Run a single job to verify the setup
python run_pipeline.py --job "Accountants"

# 4. View results in the dashboard
streamlit run dashboard.py
```

Running the pipeline writes into `results/` by default and replaces the committed files there: step 3 overwrites `results/accountant_auditor_results_auto.json`, and the dashboard then labels your run "Qwen 35B". Add `--results-dir <dir>` to write elsewhere, or restore the committed files with `git checkout -- results/`.

Or use the Makefile:

```bash
make setup       # installs deps + copies config template
make run JOB="Accountants"
make dashboard
```

Or run the dashboard in Docker. It serves at `http://localhost:7860/ai-exposure/`. To serve it at the root instead, set `BASE_URL_PATH=` (empty) in `.env`:

```bash
docker compose up -d --build
```

---

## Configuration

Configuration is resolved in priority order (highest first):

```
CLI arguments  >  environment variables (.env)  >  pipeline_config.json  >  built-in defaults
```

### pipeline_config.json

Copy `pipeline_config.example.json` → `pipeline_config.json` (never committed to git):

```json
{
  "server": {
    "ip": "10.0.0.1",
    "teacher_port": 8000,
    "student_port": 8000
  },
  "models": {
    "teacher": "meta-llama/Llama-3-8B-Instruct",
    "student": "meta-llama/Llama-3-8B-Instruct"
  }
}
```

The pipeline has three roles: **teacher** (writes tasks), **student** (the model being evaluated) and **judge** (grades answers). The judge uses the teacher's settings unless you give it its own; `pipeline_config.example.json` sets `judge_port` to 8000, so change or delete it if your teacher uses another port. Optional per-role keys in `server`: `judge_ip`, `judge_port`, `<role>_base_url`, `<role>_thinking_toggle`; and `models.judge`. Placeholder values like `YOUR_MODEL_NAME` are treated as unset.

### Environment variables

Copy `.env.example` → `.env` (also never committed). `<ROLE>` is `TEACHER`, `STUDENT` or `JUDGE`:

| Variable | Description |
|---|---|
| `EVAL_SERVER_IP` | Server IP or hostname (all roles) |
| `EVAL_SERVER_PORT` | Port (all roles) |
| `EVAL_<ROLE>_MODEL` | Model name for that role (judge defaults to teacher's) |
| `EVAL_<ROLE>_IP` / `EVAL_<ROLE>_PORT` | Role-specific server (optional) |
| `EVAL_<ROLE>_BASE_URL` | Full OpenAI-compatible base URL, e.g. `https://api.openai.com/v1`. Takes precedence over IP/port; use for hosted APIs or HTTPS |
| `EVAL_<ROLE>_API_KEY` | Sent as `Authorization: Bearer …` |
| `EVAL_API_KEY` | Shared key for every role that has no role-specific one |
| `EVAL_<ROLE>_THINKING_TOGGLE` | Send vLLM's `chat_template_kwargs.enable_thinking=false` (Qwen3). Default: on for IP/port, off for `BASE_URL` |

API keys are read only from the environment — never from `pipeline_config.json` or the command line. The judge reuses the teacher's key only when both point at the same URL.

### CLI arguments

| Argument | Description |
|---|---|
| `--server-ip` | Override server IP (all roles) |
| `--port` | Override port (all roles) |
| `--teacher-model` / `--student-model` / `--judge-model` | Override a role's model |
| `--teacher-base-url` / `--student-base-url` / `--judge-base-url` | Override a role's base URL |
| `--pipeline-config` | Use a different config file |

At startup the pipeline prints the model and URL each role will use, and stops with a specific message if a model is missing, a server can't be reached, or a key is rejected.

Example — run without editing any file:
```bash
EVAL_SERVER_IP=10.0.0.1 EVAL_TEACHER_MODEL=YOUR_MODEL_NAME EVAL_STUDENT_MODEL=YOUR_STUDENT_MODEL python run_pipeline.py --job "Budget"
# or
python run_pipeline.py --server-ip 10.0.0.1 --teacher-model YOUR_MODEL_NAME --student-model YOUR_STUDENT_MODEL --job "Budget"
```

---

## Using Your Own Model

The pipeline calls a standard OpenAI-compatible `/v1/chat/completions` endpoint. Any server that implements this API works.

### vLLM (recommended for local models)

```bash
pip install vllm
vllm serve meta-llama/Llama-3-8B-Instruct --port 8000
```

Then set `"ip": "localhost"` and `"teacher_port": 8000` in `pipeline_config.json`.

### Ollama

```bash
ollama pull llama3
ollama serve   # runs on localhost:11434
```

Set `teacher_port`, `student_port` and `judge_port` to `11434`, or pass `--port 11434`. Note: Ollama's model names use short names like `llama3`, not HuggingFace paths.

### Hosted APIs (OpenAI, OpenRouter, Together, …)

Use a base URL and an API key instead of IP/port:

```bash
export EVAL_TEACHER_BASE_URL=https://api.openai.com/v1
export EVAL_TEACHER_MODEL=YOUR_MODEL_NAME
export EVAL_TEACHER_API_KEY=sk-...
```

The vLLM thinking flag is off by default for base-URL endpoints, because many hosted APIs reject unknown request fields. If a base URL points at your own vLLM server running Qwen3, set `EVAL_<ROLE>_THINKING_TOGGLE=1`.

### Mixing hosts

Use a strong hosted model as teacher and judge, and evaluate a local student:

```bash
python run_pipeline.py --job "Budget" \
  --teacher-base-url https://api.openai.com/v1 --teacher-model YOUR_MODEL_NAME \
  --server-ip 10.0.0.1 --student-model YOUR_STUDENT_MODEL
# EVAL_TEACHER_API_KEY must be set; the judge follows the teacher
```

---

## Running the Pipeline

```bash
# Single job (interactive picker if no --job)
python run_pipeline.py
python run_pipeline.py --job "Web Developers"

# All 114 jobs
python run_pipeline.py --batch-all

# Resume interrupted run (skip completed jobs)
python run_pipeline.py --batch-all --resume

# Fill gaps in existing results (re-grade missing tasks)
python run_pipeline.py --patch-all
python run_pipeline.py --patch --job "Accountants"

# Generate job config only, skip evaluation
python run_pipeline.py --job "Accountants" --skip-eval

# Use a saved job config (from --skip-eval, or the example test_saved_config.json)
python run_pipeline.py --config test_saved_config.json
```

---

## Output Format

Each job produces `results/{job_id}_results_auto.json`. The abridged example below follows `results/accountant_auditor_results_auto.json` (five dimensions). Files written by a full run of the current code (not `--patch`) also list Autonomy and add a `judge_model` field, which the committed result files don't have:

```json
{
  "meta": {
    "job_title": "Accountants and Auditors",
    "onet_code": "13-2011.00",
    "teacher_model": "...",
    "student_model": "...",
    "total_onet_tasks": 30,
    "evaluated_tasks": 30,
    "grading_dimensions": ["Correctness", "Completeness", "Best Practices", "Domain Accuracy", "Clarity"],
    "exposure_score": {
      "overall": 4.04,
      "dimensions": {
        "Correctness": 3.60,
        "Completeness": 4.23,
        "Best Practices": 3.73,
        "Domain Accuracy": 3.77,
        "Clarity": 4.87
      },
      "evaluated_tasks": 30,
      "scale_max": 5
    }
  },
  "tasks": [
    {
      "task_id": "gen_task_01",
      "onet_source": "Original O*NET task statement",
      "task_type": "reporting",
      "user_prompt": "Full task description given to the student model",
      "reference_context": "Data / context provided alongside the task",
      "student_answer": "Model's complete answer",
      "dimension_scores": [4, 5, 4, 5, 5],
      "score": 4.6,
      "grade_reason": "Judge's critique-based explanation"
    }
  ]
}
```

`task_id` restarts in each batch (five tasks, or six in `results_gemma3_4b/`), so it isn't unique within a file.

---

## Project Structure

```
ai-exposure-eval/
├── run_pipeline.py              # Main pipeline entry point
├── utils.py                     # Shared helpers: LLM calls, config, prompt loading
├── dashboard.py                 # Streamlit analytics dashboard
├── analyze_dimensions.py        # Per-dimension and fluent-but-flawed gap analysis
├── dimension_analysis_summary.csv  # Output of analyze_dimensions.py ("default" = results/; the one "7b" row's results aren't included)
├── filter_digital_jobs.py       # Picks the 114 digital occupations (needs O*NET Work Activities.xlsx, not included)
├── run_phase2.py, run_phase2.sh, run_phase3.py  # Multi-model batch runners (of their outputs, only the partial results_1.5b_v2/ run is included)
│
├── pipeline_config.example.json # Config template (copy → pipeline_config.json)
├── pipeline_config_7b.json      # Config for a Qwen 7B student run (results not included)
├── test_saved_config.json       # Example saved job config (Accountants and Auditors)
├── .env.example                 # Env var template (copy → .env)
├── requirements.txt             # Pipeline and dashboard dependencies (figures also need seaborn)
├── constraints.txt              # Versions pinned for the dashboard image
├── Dockerfile                   # Dashboard image
├── docker-compose.yml           # Runs the dashboard on port 7860
├── Makefile                     # Convenience commands
│
├── prompt_templates/            # Prompt templates (edit to change behavior)
│   ├── generate_configuration.txt
│   ├── step1_teacher_tasks.txt
│   ├── step2_student_system.txt
│   ├── step3_grading.txt
│   └── task_selector.txt
│
├── digital_jobs.csv             # 114 pre-filtered digital occupations
├── Task Statements.xlsx         # O*NET 28.0 task statements (source of truth)
│
├── configs/                     # Cached LLM-generated job configs (auto-created)
├── results/                     # Qwen 35B run (pipeline output, auto-created)
├── results_gemma3_4b/           # Gemma 3 4B run
├── results_1.5b_v2/             # Qwen 1.5B partial run
├── data/figures/                # Blog figures and aggregated_results.csv (Autonomy 0 = not scored)
└── scripts/                     # generate_blog_figures.py
```

Prompt templates are plain `.txt` files with `{placeholder}` substitution. Edit them to change task generation style, grading strictness, or scoring criteria — no code changes needed.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The code is released under the [MIT License](LICENSE). O\*NET® information, wherever it appears in this repository, is under CC BY 4.0; see [Data Sources](#data-sources).

## Data Sources

This repository includes information from the [O\*NET 28.0 Database](https://www.onetcenter.org/db_releases.html) by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA). Used under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. O\*NET® is a trademark of USDOL/ETA. The ai-exposure-eval contributors have modified all or some of this information. USDOL/ETA has not approved, endorsed, or tested these modifications.

Where O\*NET information appears, and what changed:

- `Task Statements.xlsx`: the O\*NET 28.0 Task Statements file (19,281 task statements for 923 occupations).
- `digital_jobs.csv`: the 114 occupations that `filter_digital_jobs.py` selected using O\*NET Work Activities importance ratings, with those ratings (`Work Activities.xlsx` itself isn't included).
- `configs/`, `test_saved_config.json` and the result directories: O\*NET task statements (verbatim in `onet_tasks`; in `onet_source` the teacher model occasionally shortened, merged or reworded the statement it echoed), plus the tasks, answers and scores that models wrote from them.
- `dimension_analysis_summary.csv` and `data/figures/`: O\*NET occupation titles (and, in the summary, O\*NET-SOC codes) next to the scores.
