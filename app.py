
import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv()
def main():
    st.sidebar.title('PDF ASSISTANT')
    st.sidebar.markdown('''
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

    st.write('kgodfrey & sambutracy')

    st.header("Interact with pdf")

    # upload a PDF file
    pdf = st.file_uploader("Upload PDF Document", type='pdf')
 
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

        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key is None:
            raise ValueError("Gemini API key not found. Please make sure to set it in your .env file.")
        
        embeddings = None
        VectorStore = None

        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
            store_name = pdf.name[:-4]
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(f"{store_name}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if VectorStore:
            query = st.text_input("Ask questions about your PDF file:")
            if st.button("Ask"):
                if not query:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner('Searching for answers...'):
                        docs = VectorStore.similarity_search(query=query)
                        #st.warning("i")
                        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,convert_system_message_to_human=True)
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                        with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=query)
                        st.write(response)

                        if query:
                            st.write("Search Results:")
                            
                            #print(response)
                            #st.write("Reply: ", response["output_text"])
                            #st.warning("i")
                            #for i, doc in enumerate(docs):
                            #    st.write(f"{i + 1}. {doc['title']} - {doc['excerpt']}")
                            response = chain(
                                {"input_documents":docs, "question": query}
                                , return_only_outputs=True)

                            print(response)
                            st.write("Reply: ", response["output_text"])


if __name__ == '__main__':
    main()
