import asyncio
from langchain_chroma import Chroma  # Vector database
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Google's LLM and embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage  # Message types for chat
from langgraph.prebuilt import create_react_agent  # For creating an agent that can use tools
from langchain.tools import tool  # Decorator to create tools
from difflib import get_close_matches
import streamlit as st
import pandas as pd

def load_car_data(dataset_path):
    car_data = pd.read_csv(dataset_path)
    car_data['brand'] = car_data['brand'].str.strip().str.lower()
    car_data['model'] = car_data['model'].str.strip().str.lower()
    return car_data

api_key = "AIzaSyC2gfOZUIiX4CgiFybCKKHCY1HD9XDwpfA"
directory_path = r'/Users/charbelkadi/Desktop/COE/Fall 2024/Large Language Models - COE548/Used Car Price Prediction Dataset export 2024-12-11 15-24-57.csv'

df = load_car_data(directory_path)

@tool
def get_car_price(brand: str, model: str = None):
    """
    Fetch the price of a specific car or all cars under a brand if model is not provided.
    """
    brand = brand.strip().lower()
    if model:
        model = model.strip().lower()

        # Find closest matches for brand and model
        matched_brand = get_close_matches(brand, df['brand'].unique(), n=1, cutoff=0.7)
        if matched_brand:
            matched_brand = matched_brand[0]
            matched_model = get_close_matches(model, df[df['brand'] == matched_brand]['model'], n=1, cutoff=0.7)
            if matched_model:
                matched_model = matched_model[0]
                car = df[(df['brand'] == matched_brand) & (df['model'] == matched_model)]
                return car[['brand', 'model', 'price']].to_dict(orient='records')
            else:
                return f"No matching model found for '{model}' under brand '{brand}'."
        else:
            return f"No matching brand found for '{brand}'. Please check the input."
    else:
        # Find closest matches for brand only
        matched_brand = get_close_matches(brand, df['brand'].unique(), n=1, cutoff=0.7)
        if matched_brand:
            matched_brand = matched_brand[0]
            cars = df[df['brand'] == matched_brand]
            return cars[['brand', 'model', 'price']].to_dict(orient='records')
        else:
            available_brands = ', '.join(df['brand'].unique())
            return (f"No cars found for brand '{brand}'. "
                    f"Available brands in the dataset are: {available_brands}.")

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
        self.agent = create_react_agent(self.llm, tools=[get_car_price])
        
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
    
    
async def main():
     st.title("Gemini Chat")
     st.write("Ask anything about the documents in the vector database")

    # Initialize LLM instance if not already in session state
    # This ensures the chat model persists across page refreshes
    # Also ensures that the LLM instance is created only once
     if "llm" not in st.session_state: # session state is a dictionary that stores the state of the application and does not get reset on page refresh
        st.session_state.llm = GeminiChat()
        
    # Initialize message history in session state if not already present
    # This stores the chat history between user and AI across page refreshes
     if "messages" not in st.session_state:
        st.session_state.messages = [] # Empty list to store message history

    # Display all previous messages from session state
     for message in st.session_state.messages:
        # Create chat message UI element with appropriate type (user/assistant) and display content
        
        # Handle AI message with content (regular response)
        if isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                st.markdown(message.content)
        # Handle AI message without content (tool call)
        elif isinstance(message, AIMessage) and not message.content:
            with st.chat_message("assistant"):
                # Extract tool name and arguments from the tool call
                tool_name = message.tool_calls[0]['name']
                tool_args = str(message.tool_calls[0]['args'])
                # Display tool call details with status indicator
                with st.status(f"Tool call: {tool_name}"):
                    st.markdown(tool_args)
        # Handle tool execution result message
        elif isinstance(message, ToolMessage):
            with st.chat_message("assistant"):
                # Display tool execution result with status indicator
                with st.status("Tool result: "):
                    st.markdown(message.content)
        # Handle user message
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
    
    # Get user input from chat interface using Streamlit's chat_input widget
    # Returns None if no input is provided
     prompt = st.chat_input("Your message")

    # Get user input from chat interface. 
     if prompt:
        # Add user's message to session state history
        st.session_state.messages.append(HumanMessage(content=prompt))
        # Display user's message in chat UI
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send message to LLM and get response messages (may include tool usage)
        messages = st.session_state.llm.send_message(prompt)
        
        # Add all new messages (including tool calls) to session state history
        st.session_state.messages.extend(messages)

        # Process response messages
        for message in messages:
            # Check if message is from AI (not a tool call) and has content
            # When it is a tool call, AIMessage object is created but it has no content
            # isinstance(message, AIMessage) will skip tool outputs
            if isinstance(message, AIMessage) and message.content:
                # Display AI's regular response message
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage) and not message.content:
                # Handle AI message that contains a tool call
                with st.chat_message("assistant"):
                    # Extract tool name and arguments from the tool call
                    tool_name = message.tool_calls[0]['name']
                    tool_args = str(message.tool_calls[0]['args'])
                    # Display tool call details with status indicator
                    with st.status(f"Tool call: {tool_name}"):
                        st.markdown(tool_args)
            elif isinstance(message, ToolMessage):
                # Display the result returned from tool execution
                with st.chat_message("assistant"):
                    with st.status("Tool result: "):
                        st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                # Display user's message
                with st.chat_message("user"):
                    st.markdown(message.content)

asyncio.run(main())    