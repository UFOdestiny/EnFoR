# EnFoR

**Entailment over Frame-of-Reference-preserving parses** for spatial question answering.

Spatial descriptions state relations in different frames of reference (an object's own front, a viewer, a traveler's
heading) and often leave facings unstated. EnFoR decides whether the text determines the answer:

1. **FoR-preserving parsing.** An LLM copies each relation together with the frame it is stated in and leaves every
   unstated facing open. It converts nothing.
2. **Exact entailment.** A solver treats unstated facings (and unstated distances) as variables, checks every
   completion consistent with the text, and returns the entailed answer, or *Unknown* with every possible answer.
   When the parse cannot be read, the same model's chain-of-thought answer is used.

## Setup

```bash
pip install -e ".[vllm]"          # add ".[claude]" for Claude models
cp .env.example .env              # set model, dataset and output locations
```

Put the open models (e.g. `gpt-oss-120b`, `Gemma-4-31B`, `Llama-3.3-70B`) under `ENFOR_MODELS` and the public
benchmarks under `ENFOR_DATASETS`:

| Folder | Benchmark |
|---|---|
| `FoREST/` | FoREST (`Dataset/`, `utils/prompt.py`) |
| `RouteSurvey/data/` | Route->Survey test files |
| `RoomSpace/hf_dataset/` | RoomSpace (Hugging Face `save_to_disk`) |
| `SpatialEval/tqa/` | SpatialEval text-only parquet |
| `SpaceNLI/hf_dataset/` | SpaceNLI (Hugging Face `save_to_disk`) |

The evaluation sets derived for the paper are in `data/`: StepGame answerability (held-out, two paraphrases, free
prose, minimal pairs) and free-prose rewrites of the Route->Survey route descriptions.

## Usage

```bash
python -m enfor list                                                  # test sets
python -m enfor run --column sg_heldout --model gpt-oss-120b          # parse, CoT fallback, score
python -m enfor run --column forest_amb_f1 --model gpt-oss-120b --k 5 # 5 parses, majority of decided parses
```

`parse`, `cot` and `evaluate` run the three steps separately. Logs are written to `ENFOR_OUTPUT/enfor/<model>/` and
resume after an interruption. Open models run with vLLM at temperature 1.0 with one seed per request.

## Layout

```
enfor/
  tasks/      one module per benchmark: items, parser prompt, decision, CoT fallback, metric
  solvers/    exact solvers: facings (StepGame, Z3), frames (FoREST), route (Route->Survey),
              room_grid (RoomSpace), maze (SpatialEval), spacenli (SpaceNLI)
  pipeline.py parse -> decide -> vote -> fallback -> score
  llm.py      vLLM and Claude generation with resumable logs
data/         derived evaluation sets
```

## License

MIT
