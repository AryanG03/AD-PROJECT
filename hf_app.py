"""
hf_app.py
HuggingFace Spaces entry point (Gradio SDK - free tier).
Launches our FastAPI app on port 7860 via uvicorn,
then uses a minimal Gradio redirect so HF knows it's alive.
"""
import sys, os, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn
import gradio as gr
from app import app as fastapi_app

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="warning")

# Start FastAPI in a background thread
thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()
time.sleep(3)   # give FastAPI time to start

# Minimal Gradio interface that forwards users to the real app
with gr.Blocks(title="Neuro-CX") as demo:
    gr.HTML('''
    <div style="text-align:center; padding:2rem; font-family:sans-serif;">
      <h1 style="font-size:2rem;">🧠 Neuro-CX</h1>
      <p style="color:#666;">Neuroplasticity-inspired recommendation system</p>
      <a href="/" target="_top"
         style="display:inline-block; margin-top:1rem; padding:0.75rem 2rem;
                background:linear-gradient(135deg,#667eea,#764ba2); color:white;
                border-radius:8px; text-decoration:none; font-weight:700; font-size:1rem;">
        Open App →
      </a>
    </div>
    ''')

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
