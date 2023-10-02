import streamlit as st
import os
import openai
import pickle
#step 1 converting pdf to text
from PyPDF2 import PdfReader
#step 1 splitting data 
from langchain.text_splitter import RecursiveCharacterTextSplitter
#step 3 embeddings
from langchain.embeddings.openai import OpenAIEmbeddings;
# importing a vector store , 
from langchain.vectorstores import FAISS # you can use any from this, eg : Chroma
#step 4
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]



# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    pdf_reader = PdfReader(pdf_file)
    
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    
    return pdf_text

# Set the theme to light mode and add an emoji icon
st.set_page_config(
    page_title="pdf reader app",
    page_icon="📄",  # Emoji icon related to PDF
    layout="wide",
    # initial_sidebar_state="expanded",  # You can change this as needed
)

# Add some CSS styling to the text
st.markdown(
    """
    <style>
    .custom-heading {
    color: #34013f;
    }
    .text-box {
        padding: 20px;
        background-color: #34013f;
        color:#fff;
        border-radius: 10px;
        margin-top:20px;
        box-shadow: rgba(0, 0, 0, 0.12) 0px 1px 3px, #34013f 0px 1px 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app with a sidebar
st.markdown("<h1 class='custom-heading'>Robo Reader</h1>", unsafe_allow_html=True)

# Define content for the sidebar
with st.sidebar:
    # st.header("Sidebar Content")
    image_path = 'https://png.pngtree.com/png-vector/20230321/ourmid/pngtree-artificial-intelligence-robot-illustration-png-image_6654937.png'  # Replace with the actual path to your image file
    st.image(image_path, caption="Chat with your pdf files at ease...", use_column_width=True)
   
    # You can add widgets and other content to the sidebar here



# 1. Upload a PDF file and extracted data in pdf_text variable-----------------------------------------------------
current_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if current_pdf is not None:
    pdf_text = extract_text_from_pdf(current_pdf)
    
    # Display the extracted text---------------------------------------
    # st.header("Extracted Text from PDF")
    # st.write(pdf_text)
    
#2. splitting text into chunks with RecursiveCharacterTextSplitter..
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text=pdf_text)
    # st.write(current_pdf.name[:-4])

#3. converting text to vectors ie, by embedding , 
#   so that we can compare two and get the similiarity.
# we are using openAi's text embeddings.

    # -------> optimised embeddings = OpenAIEmbeddings(); #this is just an embedding object, we need to compute embeddings on documnets
# there are very many langchain vector store. chroma etc .
# here we are using FAISS
    # -------> optimised  VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
# Money Management , be very carefull in the above sentense will cost 
# to optimise we need to 
    store_name = current_pdf.name[:-4]
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            VectorStore = pickle.load(f)
        # st.write("embeddings loaded from disc")
    else :
        embeddings = OpenAIEmbeddings();
        VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(VectorStore,f)
    

    #Accept user question 
    query = st.text_input("Ask questions about your pdf file...")
    # st.write(query)
    if query : 
        docs = VectorStore.similarity_search(query=query)
        # st.write(docs)
        #---------we need to pass these 3 similiar results to LLM to combine.---------------------------------------
    #step 4 . feed  these 3 similiar results to LLM to combine.
        # llm = OpenAI(temperature=0,) # you can change model here, defualt davinci, set it to gpt 3.5 turbo
        llm = OpenAI(model_name="gpt-3.5-turbo",temperature=1.5)
        chain = load_qa_chain(llm=llm,chain_type="stuff");
        response = chain.run(input_documents=docs,question=query)
        # st.write(response)
# else :
    # gif_path = "load.gif"  # Replace with the actual path to your GIF file
    # st.image(gif_path, caption="Animated GIF", use_column_width=True)
        st.markdown(f"<div class='text-box'>{response}</div>", unsafe_allow_html=True)