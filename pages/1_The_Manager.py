import streamlit as st
import numpy as np
import utils
import translator

st.set_page_config(page_title="The Manager", page_icon="💼")
utils.apply_custom_css()

# --- HEADER ---
st.title("💼 The Manager")
st.markdown("**The Office Politician**")
st.caption("Paranoid, passive-aggressive, and hates clear communication.")

# --- MODEL LOADING ---
tokenizer, model = utils.load_onnx_model("model_corporate")

if tokenizer and model:
    user_input = st.text_area(
        "Paste email here:", height=150, placeholder="Per my last email... Regards."
    )

    if st.button("Decode BS", type="primary"):
        if not user_input.strip():
            st.warning("I can't analyze silence, but I can judge it.")
        else:
            # Inference
            inputs = tokenizer(
                user_input, return_tensors="pt", truncation=True, max_length=128
            )
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]

            id2label = {0: "POSITIVE", 1: "NEUTRAL", 2: "PASSIVE_AGGRESSIVE"}
            label = id2label[pred_idx]

            # Translation
            blunt_translation = translator.translate_passive_aggression(
                user_input, label, confidence
            )

            # Render Result
            utils.render_result_card(label, confidence, blunt_translation)
