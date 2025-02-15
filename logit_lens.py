"""
Implementation of the Logit Lens heatmap of layer-by-layer token predictions.
This implementation was heavily inspired by the work of the original Logit Lens
designer Nostalgebraist:
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
"""

import torch
import numpy as np
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, LabelSet
)
from bokeh.plotting import figure
from bokeh.palettes import Iridescent23, BuRd9, TolYlOrBr9
from model import model, tokenizer, get_activations

# Define the color palettes for the heatmap
allowed_palettes = {
    "BuRed": list(reversed(BuRd9)),
    "Iridescent": list(reversed(Iridescent23)),
    "Sunset": list(reversed(TolYlOrBr9))
}

def get_tokens_and_correct_outputs(inputs):
    """
    Extract token IDs and compute token strings and shifted correct outputs.
    """
    input_ids = inputs["input_ids"][0]
    n_tokens = len(input_ids)
    input_tokens = [tokenizer.decode([tid]).strip() for tid in input_ids]
    correct_outputs = [tokenizer.decode([input_ids[i+1]]).strip() for i in range(n_tokens - 1)]
    correct_outputs.append("")  # Account for last token having no next token

    return input_ids, n_tokens, input_tokens, correct_outputs

def compute_prediction_matrices(hidden_states, lm_head_weight, n_tokens):
    """
    Compute prediction, logit, and log probability matrices for each layer.
    """
    pred_matrix = []
    logit_matrix = []
    log_prob_matrix = []
    # Iterate through the model's hidden states.
    for layer_state in hidden_states:
        logits_layer = torch.matmul(layer_state, lm_head_weight.T)[0]
        layer_preds = []
        layer_logits = []
        layer_log_probs = []
        # Iterate through the token sequence.
        for j in range(n_tokens):
            token_logits = logits_layer[j]
            top_idx = torch.argmax(token_logits).item()
            top_logit = token_logits[top_idx].item()
            log_sum_exp = torch.logsumexp(token_logits, dim=0).item()
            log_prob = top_logit - log_sum_exp
            layer_preds.append(tokenizer.decode([top_idx]).strip())
            layer_logits.append(top_logit)
            layer_log_probs.append(log_prob)
        pred_matrix.append(layer_preds)
        logit_matrix.append(layer_logits)
        log_prob_matrix.append(layer_log_probs)

    return np.array(pred_matrix), np.array(logit_matrix), np.array(log_prob_matrix)

def prepare_bokeh_source(pred_matrix, logit_matrix, log_prob_matrix, final_preds,
                         n_tokens, num_layers, logit_threshold):
    """
    Prepare the cell data for the Bokeh heatmap.
    """
    xs = []
    ys = []
    cell_pred = []
    cell_logit = []
    cell_outline = []
    cell_log_prob = []
    cell_layer_num = []
    text_colors = []
    # Iterate for each layer of the model.
    for layer in range(num_layers):
        # Iterate through the token sequence.
        for token_idx in range(n_tokens):
            xs.append(str(token_idx))
            ys.append("INPUT" if layer == 0 else f"Layer {layer}")
            cell_pred.append(pred_matrix[layer, token_idx])
            logit_val = logit_matrix[layer, token_idx]
            cell_logit.append(logit_val)
            cell_outline.append(pred_matrix[layer, token_idx] == final_preds[token_idx])
            cell_layer_num.append(layer)
            cell_log_prob.append(log_prob_matrix[layer, token_idx])
            text_colors.append("black" if logit_val > logit_threshold else "white")
    # Collect and organize the source data for the heatmap.
    source = ColumnDataSource(data=dict(
        x=xs,
        y=ys,
        pred=cell_pred,
        logit=cell_logit,
        log_prob=cell_log_prob,
        layer_num=cell_layer_num,
        outline=cell_outline,
        text_color=text_colors
    ))

    return source, xs, ys

def create_heatmap_figure(source, xs, ys, input_tokens, correct_outputs, final_preds,
                          n_tokens, num_layers, selected_palette, logit_matrix):
    """
    Create and return the heatmap figure with cells, outlines, labels, and color bar.
    """
    # Define the color map.
    mapper = LinearColorMapper(palette=selected_palette,
                               low=np.min(logit_matrix), high=np.max(logit_matrix))
    # Define the Logit Lens figure.
    p = figure(title="Logit Lens Heatmap",
               x_range=[str(i) for i in range(n_tokens)],
               y_range=["INPUT"] + [f"Layer {i}" for i in range(1, num_layers)],
               height=400, width=900, tools="save,pan,box_zoom,reset")
    p.rect(x='x', y='y', width=1, height=1, source=source,
           fill_color={'field': 'logit', 'transform': mapper},
           line_color="black", line_width=0.5)

    # Assingn the source data to the figure.
    outline_source = ColumnDataSource(data=dict(
        x=[xs[i] for i in range(len(xs)) if source.data['outline'][i]],
        y=[ys[i] for i in range(len(ys)) if source.data['outline'][i]]
    ))
    p.rect(x='x', y='y', width=1, height=1, source=outline_source,
           fill_color=None, line_color="black", line_width=2)
    labels = LabelSet(x='x', y='y', text='pred', source=source,
                      text_align="center", text_baseline="middle",
                      text_color='text_color')
    p.add_layout(labels)
    p.xaxis.major_label_overrides = {str(i): token for i, token in enumerate(input_tokens)}
    p.xaxis.major_label_orientation = 0.785

    # Set up the asterisks for correct token predictions.
    top_labels = []
    for i in range(n_tokens):
        corr = correct_outputs[i]
        final_pred = final_preds[i]
        top_labels.append(f"{corr} *" if corr and (final_pred == corr) else corr)

    # Set up the top row of actual next tokens.
    top_source = ColumnDataSource(data=dict(
        x=[str(i) for i in range(n_tokens)],
        y=[num_layers + 0.3] * n_tokens,
        text=top_labels
    ))
    p.text(x='x', y='y', text='text', source=top_source,
           text_align="center", text_baseline="bottom", text_font_style="bold")

    p.y_range.range_padding = 0.2

    color_bar = ColorBar(
        color_mapper=mapper,
        label_standoff=6,
        border_line_color=None,
        location=(0,0)
    )
    p.add_layout(color_bar, 'right')

    return p

def logit_lens_heatmap(prompt, selected_palette):
    """
    Generates a heatmap for the logit lens visualization.
    """
    # Run the prompt through the model and collect activations.
    inputs, _, hidden_states, _ = get_activations(prompt)
    # Get the weight matrix for decoding the activations.
    lm_head_weight = model.lm_head.weight  # (vocab_size, hidden_dim)

    input_ids, n_tokens, input_tokens, correct_outputs = get_tokens_and_correct_outputs(inputs)
    num_layers = len(hidden_states) 

    # Get the layer-by-layer predictions for the token sequence
    pred_matrix, logit_matrix, log_prob_matrix = compute_prediction_matrices(hidden_states, lm_head_weight, n_tokens)
    final_preds = pred_matrix[-1]
    logit_threshold = (np.max(logit_matrix) + np.min(logit_matrix)) / 2

    # Get the data for the Logit Lens.
    source, xs, ys = prepare_bokeh_source(pred_matrix, logit_matrix, log_prob_matrix,
                                            final_preds, n_tokens, num_layers, logit_threshold)

    # Create the Logit Lens.
    p = create_heatmap_figure(source, xs, ys, input_tokens, correct_outputs, final_preds,
                              n_tokens, num_layers, selected_palette, logit_matrix)

    return p
