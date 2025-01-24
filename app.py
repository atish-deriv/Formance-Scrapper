import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_chain():
    """Initialize the conversation chain with Pinecone and OpenAI."""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="text-embedding-ada-002"
    )
    
    # Initialize vector store
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="formance",
        embedding=embeddings,
        text_key="content",
        namespace=""
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question"
    )
    
    # Custom prompt template
    prompt_template = """You are a friendly and knowledgeable assistant specializing in Formance documentation. Your goal is to provide comprehensive, well-structured responses that include practical examples whenever possible.
IMPORTANT GUIDELINES: THIS NEEDS TO BE STRICTLY FOLLOWED AND NOT IGNORED OR SKIPPED.
Guidelines for providing detailed responses:
1. **Accuracy First**
   - Only provide information that is explicitly present in the documentation
   - If information is not found in the context, clearly state that you cannot find it
   - Never make assumptions or infer functionality that isn't documented

2. **Start with a Clear Overview**
   - Begin with a high-level explanation of the concept
   - Highlight key points that will be covered

3. **Provide Detailed Explanations**
   - Break down complex concepts into digestible parts
   - Use clear, technical language while remaining accessible
   - Include specific examples to illustrate points
   - Reference relevant documentation sections

4. **Include Practical Examples**
   - Provide code snippets when relevant
   - Show real-world use cases
   - Explain step-by-step implementations
   - Include configuration examples if applicable

5. **Best Practices and Considerations**
   - Highlight important considerations
   - Share recommended practices
   - Mention common pitfalls to avoid
   - Discuss performance implications if relevant

6. **Related Information**
   - Connect to related concepts or features
   - Suggest relevant documentation sections
   - Mention alternative approaches if applicable

If you're unsure about any information, acknowledge the uncertainty rather than making assumptions. Aim to provide actionable insights that help users implement solutions effectively.

Context Information:
{context}

Previous Conversation:
{chat_history}

Current Question: {question}

Assistant:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        ),
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 20},
            search_type="similarity"
        ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PROMPT,
            "document_separator": "\n---\n"
        },
        return_source_documents=True,
        verbose=False
    )
    
    return chain

def generate_markdown_content(question, answer):
    """Generate markdown content for the response."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# Formance Assistant Response
Generated on: {timestamp}

## Question
{question}

## Answer
{answer}"""
    
    return md_content

# Initialize the conversation chain
if st.session_state.conversation is None:
    st.session_state.conversation = initialize_chain()

# Streamlit interface
st.title("Formance Assistant")
st.write("Ask me anything about Formance! I'll help you find the information you need.")

# Display chat history
for idx, (question, answer) in enumerate(st.session_state.chat_history):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_area("Question:", question, disabled=True, key=f"q_{idx}")
        st.text_area("Answer:", answer, disabled=True, key=f"a_{idx}")
    with col2:
        md_content = generate_markdown_content(question, answer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="💾 Download",
            data=md_content,
            file_name=f"response_{timestamp}.md",
            mime="text/markdown",
            key=f"btn_{idx}"
        )
    st.markdown("---")

# User input
query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking about your question..."):
        try:
            # Get response from chain
            response = st.session_state.conversation.invoke({
                "question": query
            })
            
            # Extract answer and sources
            answer = response.get('answer', "I apologize, but I couldn't find relevant information to answer your question.")
            sources = response.get('source_documents', [])
            
            # Add to chat history
            st.session_state.chat_history.append((query, answer))
            
            # Display current response
            st.write("### Response:")
            st.write(answer)
            
            # Display sources with proper markdown formatting
            if sources:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(sources, 1):
                        url = doc.metadata.get('url', 'Unknown')
                        st.markdown(f"### Source {i}")
                        st.markdown(f'<div style="font-size: 0.8em; word-wrap: break-word;"><a href="{url}" target="_blank">{url}</a></div>', unsafe_allow_html=True)
                        if doc.metadata.get('header'):
                            st.markdown(f"**Section:** {doc.metadata['header']}")
                        st.markdown(doc.page_content)
                        st.markdown("---")
            
            # Download button for current response
            col1, col2 = st.columns([4, 1])
            with col2:
                md_content = generate_markdown_content(query, answer)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="💾 Download",
                    data=md_content,
                    file_name=f"response_{timestamp}.md",
                    mime="text/markdown",
                    key="download_latest"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try rephrasing your question.")
