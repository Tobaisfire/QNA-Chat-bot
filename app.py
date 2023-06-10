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
# def imgen(prompt3):
    


#     model_path = r"D:\keval\study\Projects\Chatbot\Trained_model_using_LoRA.bin"
#     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
#     pipe.unet.load_attn_procs(model_path)
#     # pipe.to("cuda")

    
#     image = pipe(prompt3, num_inference_steps=30, guidance_scale=8.5).images[0]
#     return image.show()



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
    
    #  with gr.Tab("Prop2IMG"):
    #     prompt3= gr.Textbox(label="Prompt")
      
 
    
    
    #     output3 = gr.Textbox(label="Output")
    #     sr_btn3 = gr.Button("Imgen")
        
    #     sr_btn3.click(fn=imgen, inputs=prompt3, outputs=output3)
       



demo.launch()
