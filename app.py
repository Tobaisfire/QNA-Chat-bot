import gradio as gr
from QnA import *
# from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

def chat(prompt,number=1):
    result = qa(prompt,number)
    
    res = [d.page_content for d in result]

    return '\n'.join(res)
def chatbot(prompt2):
    result = chatqa(prompt2)

    return result

    





with gr.Blocks() as demo:
     with gr.Tab("Semantic_Search"):
        prompt= gr.Textbox(label="Prompt")

        number = gr.Textbox(label="Number of result")

        output = gr.Textbox(label="Output")
        sr_btn = gr.Button("Search")
        sr_btn.click(fn=chat, inputs=[prompt,number], outputs=output)

     with gr.Tab("QnA"):
        prompt2= gr.Textbox(label="Prompt")
      
 
    
    
        output2 = gr.Textbox(label="Output")
        sr_btn2 = gr.Button("chat")
        
        sr_btn2.click(fn=chatbot, inputs=prompt2, outputs=output2)

       



demo.launch()
