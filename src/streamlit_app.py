"""
Healthcare Agent Chat UI - Streamlit Application

A chat interface for interacting with the healthcare agent supervisor.
"""

import streamlit as st
import requests
import time
import os
from datetime import datetime
from typing import Dict, Any, List
import uuid

# Configuration - Use environment variable for Docker/production, fallback to localhost for development
API_URL = os.getenv("SUPERVISOR_API_URL", "http://localhost:8000")
HEALTH_ENDPOINT = f"{API_URL}/health"
QUERY_ENDPOINT = f"{API_URL}/query"

def check_api_health() -> bool:
    """Check if the API is accessible."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except:
        return False

def query_supervisor(query: str, conversation_id: str) -> Dict[str, Any]:
    """
    Send a query to the supervisor agent and return the response.
    
    Args:
        query: The user's question
        conversation_id: The conversation thread ID
        
    Returns:
        Response dictionary with 'response', 'sources_used', etc.
    """
    try:
        response = requests.post(
            QUERY_ENDPOINT,
            json={"query": query, "conversation_id": conversation_id},
            timeout=120  # Healthcare queries can take time
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The agent is taking too long to respond."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Please ensure services are running."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "api_healthy" not in st.session_state:
        st.session_state.api_healthy = check_api_health()

def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return dt.strftime('%I:%M:%S %p')
    except:
        return ""

def display_message(role: str, content: str, metadata: Dict[str, Any] = None):
    """Display a chat message with optional metadata."""
    with st.chat_message(role):
        st.markdown(content)
        
        if metadata:
            # Display metadata in an expander
            with st.expander("ℹ️ Details", expanded=False):
                # Check if we have enhanced source details
                source_details = metadata.get("source_details", [])
                
                if source_details:
                    # Display enhanced source information
                    st.markdown("**🔍 Sources Used:**")
                    for detail in source_details:
                        st.markdown(f"**• {detail['agent']}**")
                        if detail.get('description'):
                            st.caption(f"  _{detail['description']}_")
                        if detail.get('urls'):
                            st.caption(f"  📎 {len(detail['urls'])} reference(s):")
                            for url in detail['urls'][:5]:  # Limit to first 5 URLs
                                st.caption(f"  [{url}]({url})")
                            if len(detail['urls']) > 5:
                                st.caption(f"  ... and {len(detail['urls']) - 5} more")
                        st.markdown("")  # Add spacing
                else:
                    # Fallback to simple sources display
                    if sources := metadata.get("sources_used"):
                        st.markdown("**🔍 Sources:**")
                        for source in sources:
                            st.markdown(f"• {source}")
                
                # Display timing information
                st.markdown("---")
                cols = st.columns(2)
                
                if exec_time := metadata.get("execution_time"):
                    with cols[0]:
                        st.caption("**⏱️ Execution Time:**")
                        st.caption(f"{exec_time:.2f}s")
                
                if timestamp := metadata.get("timestamp"):
                    with cols[1]:
                        st.caption("**🕐 Time:**")
                        st.caption(format_timestamp(timestamp))

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Healthcare Agent Chat",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Healthcare Agent")
        st.markdown("---")
        
        # API Status
        st.subheader("System Status")
        if st.session_state.api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info("Run `./start-services.sh` to start services")
        
        if st.button("🔄 Check Connection"):
            st.session_state.api_healthy = check_api_health()
            st.rerun()
        
        st.markdown("---")
        
        # Conversation Management
        st.subheader("Conversation")
        st.caption(f"ID: `{st.session_state.conversation_id[:8]}...`")
        
        if st.button("🗑️ Clear Chat", type="primary"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
        
        st.markdown("---")
        
        # Information
        st.subheader("About")
        st.markdown("""
        This chat interface connects to the Healthcare Agent Supervisor,
        which can:
        
        - 🔬 Search PubMed for scientific literature
        - 🌐 Search web sources for health information
        - 📝 Synthesize comprehensive responses
        
        **Example queries:**
        - What are the symptoms of diabetes?
        - Latest research on COVID-19 vaccines
        - Treatment options for hypertension
        """)
        
        st.markdown("---")
        st.caption(f"Messages in conversation: {len(st.session_state.messages)}")
    
    # Main chat interface
    st.title("💬 Healthcare Chat Assistant")
    
    if not st.session_state.api_healthy:
        st.warning("⚠️ The API is not accessible. Please start the services using `./start-services.sh`")
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            metadata=message.get("metadata")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a healthcare question...", disabled=not st.session_state.api_healthy):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "metadata": None
        })
        
        # Display user message
        display_message("user", prompt)
        
        # Get response from supervisor
        with st.chat_message("assistant"):
            with st.spinner("Thinking... (This may take 30-60 seconds)"):
                response_data = query_supervisor(prompt, st.session_state.conversation_id)
            
            if "error" in response_data:
                error_msg = f"❌ **Error:** {response_data['error']}"
                st.error(error_msg)
                # Add error to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": None
                })
            else:
                # Display assistant response
                assistant_content = response_data.get("response", "No response received")
                st.markdown(assistant_content)
                
                # Display metadata
                metadata = {
                    "sources_used": response_data.get("sources_used", []),
                    "source_details": response_data.get("source_details", []),
                    "execution_time": response_data.get("execution_time"),
                    "timestamp": response_data.get("timestamp")
                }
                
                with st.expander("ℹ️ Details", expanded=False):
                    # Display enhanced source information
                    source_details = metadata.get("source_details", [])
                    
                    if source_details:
                        st.markdown("**🔍 Sources Used:**")
                        for detail in source_details:
                            st.markdown(f"**• {detail['agent']}**")
                            if detail.get('description'):
                                st.caption(f"  _{detail['description']}_")
                            if detail.get('urls'):
                                st.caption(f"  📎 {len(detail['urls'])} reference(s):")
                                for url in detail['urls'][:5]:  # Limit to first 5 URLs
                                    st.caption(f"  [{url}]({url})")
                                if len(detail['urls']) > 5:
                                    st.caption(f"  ... and {len(detail['urls']) - 5} more")
                            st.markdown("")  # Add spacing
                    else:
                        # Fallback to simple sources display
                        st.markdown("**🔍 Sources:**")
                        for source in metadata["sources_used"]:
                            st.markdown(f"• {source}")
                    
                    # Display timing information
                    st.markdown("---")
                    cols = st.columns(2)
                    
                    with cols[0]:
                        st.caption("**⏱️ Execution Time:**")
                        st.caption(f"{metadata['execution_time']:.2f}s")
                    
                    with cols[1]:
                        st.caption("**🕐 Time:**")
                        st.caption(format_timestamp(metadata["timestamp"]))
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                    "metadata": metadata
                })

if __name__ == "__main__":
    main()
