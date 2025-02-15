"""
Definition for the Holoviz Panel user interface.
"""

import panel as pn
from logit_lens import logit_lens_heatmap, allowed_palettes
from data import get_prompt_data

# Initializing the Panel extension
pn.extension(sizing_mode="stretch_width", template="fast")

# CSS styling
pn.config.raw_css = ["""
.bk-btn {
    background-color: black !important;
    border-color: black !important;
    color: white !important;
}
"""]

# Load the prompt data
df_prompts = get_prompt_data()
df_prompts["label"] = df_prompts["known_id"].astype(str) + ": " + df_prompts["subject"]

# *** Prompt Widgets ***
prompt_select = pn.widgets.Select(
    name="Select Prompt",
    options=df_prompts["label"].tolist(),
    value=df_prompts["label"].iloc[0],
    width=400
)
prompt_input = pn.widgets.TextAreaInput(
    name="Prompt",
    value=df_prompts["prompt"].iloc[0],
    width=400
)

def update_prompt(event):
    """
    Prompt interactivity.
    """
    selected = prompt_select.value
    row = df_prompts[df_prompts["label"] == selected].iloc[0]
    prompt_input.value = row["prompt"]
    answer_output.object = "Click to see the correct answer."
    show_answer_button.name = "Show Correct Answer"

prompt_select.param.watch(update_prompt, 'value')

# *** Correct Answer Button Widget ***
show_answer_button = pn.widgets.Button(
    name="Show Correct Answer",
    button_type="primary",
    width=300
)
answer_output = pn.pane.Markdown(
    "Click to see the correct answer.",
    width=300
)

def toggle_answer(event):
    """
    Correct answer interactivity.
    """
    selected = prompt_select.value
    row = df_prompts[df_prompts["label"] == selected].iloc[0]
    if "Correct Answer:" in str(answer_output.object):
        answer_output.object = "Click the button to see the correct answer."
        show_answer_button.name = "Show Correct Answer"
    else:
        answer_output.object = f"**Correct Answer:** {row['attribute']}"
        show_answer_button.name = "Hide Answer"

show_answer_button.on_click(toggle_answer)

# *** Color Palette Selection Widget ***
palette_select = pn.widgets.Select(
    name="Palette",
    options=list(allowed_palettes.keys()),
    value="BuRed",
    width=300
)


update_logit_button = pn.widgets.Button(name="Update Logit Lens Heatmap", button_type="primary", width=250)

logit_heatmap_pane = pn.pane.Bokeh(height=450, sizing_mode="stretch_width")

explanation_pane = pn.pane.Markdown("", width=600, styles={"font-size": "16px"})

def update_logit_heatmap(event):
    """
    Logit Lens interactivity.
    """
    prompt = prompt_input.value
    selected_palette = allowed_palettes[palette_select.value]
    p = logit_lens_heatmap(prompt, selected_palette=selected_palette)
    logit_heatmap_pane.object = p

    explanation_text = """
    ### Logit Lens Explanation:
    - **Rows**: The bottom row labeled "INPUT" shows the original tokens.
      Each subsequent row represents a layer's next-token prediction. The final layer's
      predictions are used for cell outlining and top label marking.
    - **Columns**: Represent token positions in the input sequence.
    - **Colors**: Represent the confidence (logit value) of the model's prediction.
      Colors lower on the spectrum indicate higher confidence.
    - **Asterisk**: Added to the correct output when the model's top guess matches.
    - **Outlines**: Cells are outlined when their top guess matches the final
      top guess.
    
    **NOTE**: Language models work on tokens which can be words, sub-word units, or punctuation.
    """
    explanation_pane.object = explanation_text

update_logit_button.on_click(update_logit_heatmap)

# Layout for the answer widgets.
answer_group = pn.Column(
    pn.Row(answer_output, margin=(-12,0,0,0)),
    pn.Row(show_answer_button, margin=(-20,0,0,0)),
)

# Layout for the prompt widgets.
prompt_row = pn.Row(
    prompt_input,
    answer_group,
    align='start',
    width=800
)

# Row for the prompt and color palette controls.
controls = pn.Row(
    prompt_select,
    palette_select
)

# Column for overall Logit Lens tab.
logit_tab = pn.Column(
    pn.pane.Markdown("## Logit Lens Transformer Token Prediction Heatmap"),
    controls,
    prompt_row,
    update_logit_button,
    logit_heatmap_pane,
    explanation_pane,
    sizing_mode="stretch_width"
)

# Tab left in because I plan to extend this project to include more transformer
# visualizations in future personal work.
tabs = pn.Tabs(
    ("Logit Lens", logit_tab)
)

# Dashboard definition.
dashboard = pn.Column(
    pn.pane.Markdown("# GPT-2 Internals Visualization Dashboard"),
    tabs,
    sizing_mode="stretch_width",
    margin=(0,100)
)

if __name__ == "__main__":
    pn.serve(dashboard, show=True)
