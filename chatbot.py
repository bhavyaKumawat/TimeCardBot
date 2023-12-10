import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
import textwrap
import torch
from huggingface_hub import login

# login('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')
login(st.secrets["huggingfacetoken"])

loader = UnstructuredFileLoader('Time Cards.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
vectorstore = FAISS.from_documents(text_chunks, embeddings)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             load_in_8bit=True,
                                             # load_in_4bit=True
                                             )

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=1024,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=True,
                                    retriever=vectorstore.as_retriever())


def generate_response(query):
    result = chain({"query": query}, return_only_outputs=True)
    wrapped_text = textwrap.fill(result['result'], width=500)
    return wrapped_text


st.title("Time Card Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

response = generate_response(prompt)
with st.chat_message("assistant"):
    st.markdown(response)
st.session_state.messages.append({"role": "assistant", "content": response})
