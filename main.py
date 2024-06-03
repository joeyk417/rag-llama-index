from dotenv import load_dotenv
import os
from pdf import PDFIndexer
from llama_index.core import Settings
from llama_index.llms.replicate import Replicate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer
from prompts import system_prompt, query_wrapper_prompt
import datetime


load_dotenv()

llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
    is_chat_model=True,
)

# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

#agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

pdf_indexer = PDFIndexer("australia", Settings.llm, Settings.embed_model)
australia_engine = pdf_indexer.index_pdf(os.path.join("data", "Australia.pdf"))

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    start_time = datetime.datetime.now()
    
    result = australia_engine.query(prompt)
    
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    
    print(result)
    print("Execution time:", execution_time.total_seconds())
    
