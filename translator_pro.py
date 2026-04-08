import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu
from gtts import gTTS
import os
import torch

# --- PREPARATION ---
if not os.path.exists("models"): os.makedirs("models")
if not os.path.exists("static"): os.makedirs("static")

# --- MODEL LOADING (Manual Method) ---
model_name = "Helsinki-NLP/opus-mt-en-fr"
print("Loading model components manually...")

# This loads the model and tokenizer directly into your models folder
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./models")

def translate_logic(text):
    # 1. Tokenization: Turn text into numbers the AI understands
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # 2. Generation: The model predicts the French words
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    
    # 3. Decoding: Turn numbers back into French words
    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return result

def master_process(text, reference_text):
    if not text.strip():
        return "Please enter text", "0", None

    # Perform Translation
    res = translate_logic(text)
    
    # Calculate BLEU Score (Accuracy)
    if reference_text.strip():
        ref_tokens = [reference_text.lower().split()]
        cand_tokens = res.lower().split()
        score = sentence_bleu(ref_tokens, cand_tokens)
        accuracy = f"BLEU Score: {score:.4f}"
    else:
        accuracy = "N/A (Reference required)"
    
    # Audio Synthesis (Save to static folder)
    audio_path = os.path.join("static", "output.mp3")
    tts = gTTS(res, lang='fr')
    tts.save(audio_path)
    
    return res, accuracy, audio_path

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Advanced Neural Machine Translation")
    gr.Markdown("Direct Sequence-to-Sequence (Seq2Seq) Transformer Implementation")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="English Input", lines=3)
            ref_input = gr.Textbox(label="Human Reference (for Accuracy Test)")
            btn = gr.Button("Translate & Evaluate", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="French AI Output")
            output_score = gr.Label(label="Model Accuracy")
            output_audio = gr.Audio(label="AI Voice Pronunciation")

    btn.click(master_process, inputs=[input_text, ref_input], outputs=[output_text, output_score, output_audio])

if __name__ == "__main__":
    print("Launching UI...")
    demo.launch()