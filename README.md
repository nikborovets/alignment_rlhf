trl chat --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct --device cpu

uv pip install -e ".[dev]"
uv sync --extra dev
-
