# STAT 386: Useful Data Apps with Streamlit

Two 50-minute lessons connecting reproducible data science to exploration by people who do not write Python. Quarto builds a single teaching website in `docs/`, with a separate slide deck for each day.

## Start here

Extract this folder, open it in Positron, and run from its root:

```bash
uv sync
uv run streamlit run app.py
```

The full app includes genuine precomputed Nomic v1.5 embeddings for the same 52 selected kickers as the earlier text EDA lesson. No Ollama installation or API key is required for the main lessons or the hosted app. The first PHATE fit can take longer than later interactions.

Keep the terminal running. Stop it with Ctrl+C before launching another checkpoint on the same port. These scripts run as Streamlit apps from the terminal, rather than as notebook cells.

## Teaching route

| File | Role |
|---|---|
| `day1_slides.qmd` | Purpose, design, Streamlit motivation, execution model, and basic syntax |
| `day2_slides.qmd` | Conference explorer, caching, and deployment |
| `demo.qmd` | Step-by-step code and expected results |
| `day1/01_basic.py` | Small filter, count, chart, and table |
| `day1/02_execution.py` | Terminal/browser output and a Session State counter |
| `day2/app_live.py` | Runnable starting point with TODOs |
| `day2/01_map.py` | Familiar analysis displayed without caching |
| `day2/02_filters.py` | Adds filtering and exposes repeated PHATE fits |
| `day2/03_cached.py` | Caches loading and the PHATE computation |
| `day2/04_cache_lab.py` | Transparent one-second arithmetic example for cache predictions |
| `day2/05_local_embedding.py` | Optional real cached Ollama call |
| `app.py`, `pipeline.py` | Completed app, loading validation, and cached PHATE |
| `activities/practice.ipynb` | Separate student exercises, keeping instructor apps focused |
| `instructor/guide.qmd` | Timing, preparation, cut points, and suggested answers |
| `deployment.qmd` | GitHub, uv, and Community Cloud walkthrough |
| `cheatsheet.qmd` | Updated reference organized around the two lessons |
| `reference/Streamlit_Cheatsheet.pdf` | Printable version of the reference |

Read the instructor guide before class. The final app's similarity ranking, downloads, and extra filters are extensions, not all required live coding.

## Data and model provenance

`data/talks.csv` is the same selected teaching sample used in the earlier text EDA materials. Source notes are in `data/SOURCE_NOTES.md`. Every record retains an official source URL. Only the kicker text enters the embedding model.

The supplied `talk_embeddings.npz` contains **genuine Nomic model inference**, including ordered talk IDs and kicker texts for alignment checks. The supplied artifact uses the official Nomic **quantized ONNX** export, attention-masked mean pooling, layer normalization, and L2 normalization over all 768 coordinates. This differs from the earlier lesson's Ollama inference backend, so exact vectors and PHATE layouts can differ. See `data/embedding_provenance.json` for the exact model revision, file hashes, and computation details. No synthetic vectors or fabricated model results appear in the app.

To use the vectors you saved in the prior lesson instead, run:

```bash
uv run python scripts/prepare_data.py --import-embeddings path/to/talk_embeddings.npz
```

To regenerate with local Ollama, start Ollama, then run:

```bash
ollama pull nomic-embed-text:v1.5
uv run python scripts/prepare_data.py --from-ollama
```

The optional single-message cache demo also requires that local Ollama setup:

```bash
uv run streamlit run day2/05_local_embedding.py
```

To reproduce the supplied ONNX artifact specifically, run:

```bash
uv run --group preparation python scripts/rebuild_onnx_embeddings.py
```

That instructor-only route downloads about 137 MB of official model weights into a temporary cache outside the project. Its additional packages are optional and are not needed by the deployed app. It pins the recorded model revision. The supplied `uv.lock` records versions for both the main app and preparation group.

Saved outputs keep the analysis available across restarts. Streamlit caches speed up repeated work at runtime. Editing a source file without regenerating matching model outputs is an error, not a cache policy.

## Website and slides

Install [Quarto](https://quarto.org/docs/get-started/) separately:

```bash
quarto preview
quarto render
```

PDF copies of both decks are included in `reference/`. For a fresh PDF after editing, open a rendered deck with `?print-pdf` added to its URL, then print to PDF in your browser.

Quarto generates `docs/` without running app code. Open `docs/index.html` or either rendered slide deck to read the materials. For presenting, use `quarto preview` or a hosted website. Press S in a deck for speaker notes and F for full screen.

Commit the source and `docs/` output. In GitHub Pages settings, choose Deploy from a branch, your branch, and `/docs`. Community Cloud hosts the running app separately.

## Sources

- [Streamlit app model](https://docs.streamlit.io/get-started/fundamentals/summary)
- [Client-server architecture](https://docs.streamlit.io/develop/concepts/architecture/architecture)
- [Caching](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Ollama embedding endpoint](https://docs.ollama.com/api/embed)

Reference links and deployment steps were reviewed November 23, 2026. Use `uv sync --locked` when you want uv to fail rather than update the supplied lockfile.
