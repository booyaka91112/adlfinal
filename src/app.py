import gradio as gr

demo = gr.load("jimboHsueh/jimboHsuehfinal_project_mBart", src="models")
demo.launch(share=True)
