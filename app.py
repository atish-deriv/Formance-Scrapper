import streamlit as st
import os
import json
import time
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

def initialize_validation_chain():
    """Initialize the validation chain with OpenAI."""
    # Create validation chain
    validation_chain = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.1,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    return validation_chain

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
    prompt_template = f"""You are a friendly and knowledgeable assistant specializing in Formance documentation. Your goal is to provide comprehensive, well-structured responses that include practical examples whenever possible.
IMPORTANT GUIDELINES: THIS NEEDS TO BE STRICTLY FOLLOWED AND NOT IGNORED OR SKIPPED.

DATA PROCESSING GUIDELINES (STRICTLY FOLLOW THESE):

1. UNDERSTAND THE QUESTION AND THE CONTEXT PROVIDED.
2. ALWAYS ANSWER THE QUESTION FROM THE CONTEXT PROVIDED.
3. IF THE INFORMATION IS MISSING IN THE CONTEXT STRICTLY STATE THAT YOU CANNOT FIND THE INFORMATION.
4. NEVER MAKE ASSUMPTIONS OR INFER FUNCTIONALITY THAT IS NOT IN THE CONTEXT.
5. WHEN YOU ARE PROVIDING DETAILS ABOUT A COMMAND OR API MAKE SURE THE PARAMETERS ARE CORRECTLY SPECIFIED AND THE COMMAND OR API IS CORRECTLY SPECIFIED.
6. BEFORE PROVIDING ANY COMMAND OR API IN THE OUTPUT USE CHAIN OF THOUGHTS AND VARIFY THE COMMAND IS CORRECT.

Guidelines for providing detailed responses:
1. **Accuracy First**
   - Only provide information that is explicitly present in the documentation
   - If information is not found in the context, clearly state that you cannot find it
   - Never make assumptions or infer functionality that isn't documented
   - Command and API validation: After constructing a command or API description, review it to ensure all flags and parameters align with the context.

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
{{context}}

Previous Conversation:
{{chat_history}}

Current Question: {{question}}

Assistant:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        ),
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 10},
            search_type="similarity",
            score_threshold=0.75
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
st.markdown("<small> ***Note:*** This is a beta version and the responses may not be 100% accurate. Please use with caution. Current version is 0.2.0</small>", unsafe_allow_html=True)

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
    try:
        # Step 1: Initial response generation
        with st.spinner("🔍 Searching relevant documentation..."):
            # Get relevant documentation
            response = st.session_state.conversation.invoke({
                "question": query
            })
            
        with st.spinner("✍️ Drafting initial response..."):
            # Extract initial answer
            initial_answer = response.get('answer', "I apologize, but I couldn't find relevant information to answer your question.")
            sources = response.get('source_documents', [])
        
        # Step 2: Technical validation
        with st.spinner("🔍 Validating technical details..."):
            # Initialize validation chain if not exists
            if 'validation_chain' not in st.session_state:
                st.session_state.validation_chain = initialize_validation_chain()
            
            # Validation prompt
            validation_prompt = f"""You are a technical validation agent for Formance documentation. Your task is to validate technical details and provide a properly formatted response.

VALIDATION RULES:
1. COMMAND VALIDATION (CRITICAL):
   - Check every command against the source documentation
   - Only allow command flags and parameters that are EXPLICITLY shown in the documentation
   - Remove any parameters or options that are not present in the source documentation
   - If a command is modified, add a note explaining what was removed and why

2. API VALIDATION (CRITICAL):
   - Verify all API endpoints exactly match the documentation
   - Validate all parameters and their types
   - Ensure request/response formats are accurate

3. TECHNICAL DETAILS (CRITICAL):
   - Verify configuration settings against documentation
   - Validate technical specifications and requirements
   - Check environment variables and their usage

RESPONSE FORMAT (CRITICAL - MUST FOLLOW EXACTLY):

# Overview
A clear, high-level explanation of the concept or task.

# Detailed Explanation
Step-by-step breakdown of how the feature or process works.

# Implementation
Technical details with commands in code blocks:
```bash
# Example with real values:
command --flag value
```

# Example Usage
```bash
# Real-world example with actual values
command --flag=example-value
```

# Best Practices
Important considerations and recommendations.

# Additional Information
Related concepts and references.

CRITICAL RULES:
- Use proper markdown headers with #
- All commands must be in ```bash code blocks
- Include example usage with real values
- Only include parameters explicitly shown in documentation
- Never explain corrections or validation
- Maintain consistent formatting throughout

1. **Accuracy First**
   - Only provide information that is explicitly present in the documentation
   - If information is not found in the context, clearly state that you cannot find it
   - Never make assumptions or infer functionality that isn't documented
   - Command and API validation: After constructing a command or API description, review it to ensure all flags and parameters align with the context.

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


IMPORTANT: Never add explanatory notes about corrections or mention validation.

Question: {query}
Initial Answer: {initial_answer}
Sources: {[doc.page_content for doc in sources]}

IMPORTANT: 
- Only include commands, parameters, and technical details that are EXPLICITLY shown in the source documentation
- Provide a complete, final response that follows the format guidelines
- Never mention that you're validating or correcting anything
- Focus on delivering accurate, well-structured information"""

            # Get validated answer
            validation_response = st.session_state.validation_chain.invoke(validation_prompt)
            final_answer = validation_response.content
        
        # Step 3: Format and display results
        with st.spinner("📝 Finalizing response..."):
            # Add to chat history
            st.session_state.chat_history.append((query, final_answer))
            
            # Create a container for the response header
            response_container = st.empty()
            response_container.markdown("### Response:")
            
            # Create a container for the streaming text
            text_container = st.empty()
            
            # Stream the response character by character
            response_text = ""
            for char in final_answer:
                response_text += char
                text_container.markdown(response_text)
                time.sleep(0.01)  # Adjust speed as needed
            
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
                md_content = generate_markdown_content(query, final_answer)
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
