
# Getting Started

## Prerequisites

- Python version supported by the project
- Git
- uv

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd bank_churn_predictions
```

## 2. Install dependencies

```bash
uv sync
```

## 3. Prepare the dataset

Place the original dataset in the location expected by the project.

Check the project configuration for the expected file path.

## 4. Run the analysis

Open the notebooks in the `notebooks/` directory and execute them in the documented order.

## 5. Run the tests

```bash
uv run pytest
```

## Troubleshooting

If a command fails, verify that you are running it from the project root and that the dependencies have been installed.