import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Sidebar contents
with st.sidebar:
    st.title('PDF ASSISTANT')
    st.markdown('''
    ## About
    Welcome to the PDF Assistant! This app utilizes LangChain and OpenAI's LLM model to provide an interactive chat interface for querying information from PDF documents.
    
    Learn more about the technologies used:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')

     # File Upload Component
    st.write("## Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file is not None:
        st.success('PDF file uploaded successfully!')
    
    st.write("## Settings")
    confidence_threshold = st.slider("Response Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    st.write(f"Current Threshold: {confidence_threshold}")

    # Feedback Mechanism
    st.write("## Feedback & Support")
    st.write("Have feedback or need help? Let us know!")
    st.write("[Feedback Form](https://forms.gle/6ibXUXL91Tcrs8AV9)")

    add_vertical_space(5)
    st.write('kgodfrey & sambutracy')

load_dotenv()
def main():
    st.header("Interact with pdf")

    # upload a PDF file
    pdf = st.file_uploader("Upload PDF Document", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(f"Uploaded PDF: {pdf.name}")
        st.write(f"Number of Pages: {len(pdf_reader.pages)}")
        
        # Display a preview of the PDF content
        preview_text = ""
        for page in pdf_reader.pages[:3]:  # Display content from the first few pages
            preview_text += page.extract_text()
        st.write("Preview:")
        st.text(preview_text[:500])

        # Page navigation controls
        page_number = st.number_input("Go to Page", min_value=1, max_value=len(pdf_reader.pages), value=1)
        selected_page = pdf_reader.pages[page_number - 1]
        st.write(f"Page {page_number}:")
        st.text(selected_page.extract_text())

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OpenAI API key not found. Please make sure to set it in your .env file.")
        
        embeddings = None
        VectorStore = None

        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
            # # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            # st.write(chunks)
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # # st.write('Embeddings Loaded from the Disk')s
            else:
                ##embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if VectorStore:
    
        # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
            # st.write(query)
            if st.button("Ask"):
                if not query:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner('Searching for answers...'):
                        # Processing query and generating responses
                        docs = VectorStore.similarity_search(query=query, k=3)
                        response = chain.run(input_documents=docs, question=query)
            
                        llm = OpenAI()
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                        with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=query)
                            print(cb)
                        st.write(response)

            if query:
                # Display search results in a table or list format
                st.write("Search Results:")
                for i, doc in enumerate(docs):
                    st.write(f"{i + 1}. {doc['title']} - {doc['excerpt']}")
            
 
if __name__ == '__main__':
    main()