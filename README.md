# The Energy Cost of Accuracy Across Scale in Large Language Model Agents

Code, data and analysis scripts for the manuscript “The Energy Cost of Accuracy Across Scale in Large Language Model Agents” (under review).

The study compares nine open-weight LLMs (2–3B, 7–8B and 70B+ parameters, 4-bit NF4 quantization) inside a
confidence-triggered multi-turn inference loop on three tasks (AG News classification, CNN/DailyMail
summarization, GSM8K mathematical reasoning) at three confidence thresholds, with per-sample energy measured
by CodeCarbon on a single NVIDIA RTX PRO 6000 Blackwell GPU (Google Colab).

## Repository layout

```
notebooks/
  01_main_experiment_v4.ipynb      main run: 9 models x 3 tasks x 3 thresholds x 200 samples (GPU)
  02_component_level_run.ipynb     component-level run: generation / self-evaluation measured separately,
                                   50 samples x 3 repeats (GPU)
  03_reanalysis.ipynb              re-scoring, paired tests, token-normalized energy, calibration metrics (CPU)
scripts/
  04_judge_context_sensitivity.ipynb   judge shown 1,500 vs 3,000 characters, summarization (GPU)
  make_figures.py                  Figures 1-5 and Tables 1-4 from the result CSVs
results/
  main_run/
    agentic_energy_v4_threshold5.csv
    agentic_energy_v4_threshold7.csv   primary run (tau = 7)
    agentic_energy_v4_threshold9.csv
    model_info.json                    loaded model ids, quantization config, parameter counts
    env_info.json                      library and CUDA versions
  component_run/
    components.csv                     one row per model call (gen1, judge1, gen2, ...)
    judge_context_run/
      judge_context.csv                  one row per model, sample and judge-context condition
      judge_context_summary.csv          per-model comparison (Section 4.7)
  analysis_A/                          outputs of notebook 03 (tables used in the paper)
figures/                               publication figures (PNG 300 dpi + PDF)
```

## Result file columns (main run)

| column | meaning |
|---|---|
| model, model_category, task, threshold, sample_idx | experimental cell |
| n_turns | number of turns executed (1–3) |
| confidence_history | JSON list of self-reported scores per turn |
| energy_kwh, gpu_energy_kwh, cpu_energy_kwh, ram_energy_kwh | CodeCarbon energy for the whole sample |
| emissions_kg_co2eq, elapsed_sec, timestamp | as recorded |
| final_output, per_turn_outputs, judge_raw_history | generated text of every turn and raw judge replies |
| n_new_tokens_history, truncated_any | generated tokens per turn; 1 if any turn hit the token limit |
| parse_failed_any, n_parse_failed | confidence-score parsing failures (none occurred) |
| correct / rougeL | task metric of the final output |

## Reproducing

1. Open `notebooks/01_main_experiment_v4.ipynb` in Google Colab with a GPU runtime, add a Hugging Face token
   as the Colab secret `HF_TOKEN` (read access to the gated Llama, Gemma and Mistral repositories is required),
   mount Google Drive and run all cells. Results are written to Drive and the run is resume-safe.
2. Run `notebooks/02_component_level_run.ipynb` the same way for the component-level measurements.
3. Run `notebooks/03_reanalysis.ipynb` (CPU is sufficient) and then `scripts/make_figures.py` to regenerate
   all tables and figures from the CSV files.

Software versions used for the reported runs are listed in `results/main_run/env_info.json`.

## Citation

If you use this code or data, please cite the manuscript (citation to be added upon publication).

## License

Code: MIT. Result data: CC BY 4.0. The underlying datasets (AG News, CNN/DailyMail, GSM8K) and models retain
their own licenses.
