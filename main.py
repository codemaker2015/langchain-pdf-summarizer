import gradio as gr
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

import os
import openai
os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0)

def summarize_pdf(path):
    summary = ""
    try:
        loader = PyPDFLoader(path.name)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        prompt_template = """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
    except:
        summary = "Something went wrong. \nPlease try with some other document."
    return summary
    
def upload_file(file):
    return file.name

def main():
    global input_pdf_path
    with gr.Blocks() as demo:
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to Upload a File", file_types=["pdf"])
        upload_button.upload(upload_file, upload_button, file_output)

    output_summary = gr.Textbox(label="Summary")

    interface = gr.Interface(
        fn=summarize_pdf,
        inputs=[upload_button],
        outputs=[output_summary],
        title="PDF Summarizer",
        description="",
    )
    
    interface.launch()

if __name__ == "__main__":
    main()
