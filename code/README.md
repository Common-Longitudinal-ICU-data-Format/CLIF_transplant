 ## Code directory

Open your terminal and navigate to the project directory, then run the following:

### Using uv (recommended)

```bash
cd [path/to/your/local_project]

# Install dependencies (if not already done)
uv sync

# Run the cohort identification script
uv run marimo run code/01_cohort_identification.py

# Or edit the script interactively
uv run marimo edit code/01_cohort_identification.py
```

### Using traditional venv

```bash
cd [path/to/your/local_project]

./setup_env.sh

marimo run code/01_cohort_identification.py
```

