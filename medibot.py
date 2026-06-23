import os
import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Vector DB path
DB_FAISS_PATH = "vectorstore/db_faiss"


# Load Vector Store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# Custom Prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


# Main App
def main():

    st.set_page_config(page_title="MediBot")

    st.title("🩺 MediBot - AI Medical Assistant")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display old messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User Input
    prompt = st.chat_input("Ask your medical question...")

    if prompt:

        # Show User Message
        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say that you don't know.
Don't try to make up an answer.

Only answer from the provided context.

Context:
{context}

Question:
{question}

Start the answer directly.
"""

        try:

            # Load Vector DB
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load vector database")
                return

            # Load LLM
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            # QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt(
                        CUSTOM_PROMPT_TEMPLATE
                    )
                }
            )

            # Get Response
            response = qa_chain.invoke(
                {
                    "query": prompt
                }
            )

            result = response["result"]

            source_documents = response["source_documents"]

            result_to_show = (
                result
                + "\n\n---\n### Source Documents:\n"
                + str(source_documents)
            )

            # Show Assistant Message
            st.chat_message("assistant").markdown(result_to_show)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result_to_show
                }
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()