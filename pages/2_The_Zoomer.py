import streamlit as st
import numpy as np
import utils
import translator

st.set_page_config(page_title="The Zoomer", page_icon="🧢")
utils.apply_custom_css()

# --- HEADER ---
st.title("🧢 The Zoomer")
st.markdown("**The Intern with no filter**")
st.caption("Patched with Twitter/Reddit data.")

# --- MODEL LOADING ---
tokenizer, model = utils.load_onnx_model("model_universal")

if tokenizer and model:
    user_input = st.text_area(
        "Paste chat/tweet here:", height=150, placeholder="Great job!"
    )

    if st.button("Vibe Check"):
        if not user_input.strip():
            st.warning("Bruh. Type something.")
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
