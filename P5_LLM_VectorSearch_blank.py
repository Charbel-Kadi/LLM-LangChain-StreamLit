# Import required modules
from langchain.vectorstores import Chroma  # Vector database
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Google's LLM and embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from langgraph.prebuilt import create_react_agent  # For creating an agent that can use tools
from langchain.tools import tool  # Decorator to create tools

# API key for Google services
api_key="AIzaSyCHkj8cb_5axHLHx5P16zadU2cj3h7fUE8"

# Tool definition for adding numbers
@tool
def add_two_numbers(a: int, b: int) -> str:
    """
    Adds two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        str: The sum of the two numbers
    """
    # Convert result to string since LLM expects string output
    return str(a + b)

# Tool definition for searching vector database
@tool
def search_vector_db(query: str) -> str:
    """
    Search the vector database for documents similar to the query.
    Args:
        query (str): The search query string to find relevant documents
    Returns:
        str: A concatenated string of the top 5 most similar document contents found in the vector database
    """
    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Initialize/connect to vector database
    vector_store = Chroma(
        collection_name = "embeddings",
        embedding_function = embeddings,
        persist_directory = "./vector_db",
        collection_metadata = {"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Debug print
    print("Searching the vector database for: ", query)
    
    # Perform similarity search and get top 5 results
    result = vector_store.similarity_search(query = query, k = 5)
    # Combine all document contents into single string
    result_str = "\n\n\n".join([doc.page_content for doc in result])
    
    return result_str

# Main chat class that uses Gemini LLM
class GeminiChat:
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.0):
        """
        Initialize GeminiChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gemini-pro".
            temperature (float): The temperature to use. Default is 0.0.
        """
        # Store API key
        self.api_key=api_key
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key, 
            temperature=temperature
        )
        
        # Create agent with both tools available
        self.agent = create_react_agent(self.llm, tools=[add_two_numbers, search_vector_db])
        
        # Initialize conversation history
        self.messages = []
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user message to history
        self.messages.append(HumanMessage(content=message))
        
        # Store current history length to identify new messages later
        history_length = len(self.messages)
        
        # Get response from agent, including any tool usage
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        
        # Extract only the new messages from this interaction
        new_messages = self.messages[history_length:]

        return new_messages 

# Create chat instance
chat = GeminiChat()

# Example query using vector database search
messages = chat.send_message("Give me examples on language ambiguity based on the documents in the vector database")

# Print all messages, handling both regular messages and tool calls
for message in messages:
    if message.content:
        print(type(message), message.content)
    else:
        print(type(message), message.tool_calls[0])
