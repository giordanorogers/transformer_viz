# GPT-2 Visualization Dashboard

An interactive visualization tool for exploring GPT-2's internal token predictions across layers.

## Features

- Logit Lens heatmap showing layer-by-layer token predictions
- Interactive prompt selection from curated dataset
- Token-by-token prediction visualization
- Multiple color palette options

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:

```bash
python -m frontend
```

Then open your browser to `http://localhost:5006` or whatever server the fronted
is running on (visible in the terminal).

## Project Structure

- `frontend.py`: Panel dashboard UI
- `logit_lens.py`: Logit Lens visualization implementation
- `model.py`: GPT-2 model handling
- `data.py`: Dataset loading utilities
- `known_1000.json`: Curated prompt dataset

## Credits

Logit Lens visualization inspired by [Nostalgebraist's work](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
