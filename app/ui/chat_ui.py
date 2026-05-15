import streamlit as st
import uuid
import logging
from typing import Dict, Any
import config

logger = logging.getLogger(__name__)

class ChatUI:
    """Chat user interface using Streamlit."""
    
    def __init__(self, rag_service, metrics_manager=None):
        """
        Initialize the chat UI.
        
        Args:
            rag_service: RAG service for processing queries
            metrics_manager: Optional metrics manager for tracking stats
        """
        self.rag_service = rag_service
        self.metrics_manager = metrics_manager
        
        # Initialize session state if needed
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state for chat history."""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if 'retrieved_docs' not in st.session_state:
            st.session_state.retrieved_docs = {}
        if 'current_retrieval_id' not in st.session_state:
            st.session_state.current_retrieval_id = None
    
    def render_header(self):
        """Render the application header."""
        st.header("PDF Chatbot")
        
    def render_sidebar(self):
        """Render the sidebar with options."""
        with st.sidebar:
            st.title("Options")
            
            # Retrieval options
            use_reranker = st.checkbox("Use Reranker", value=config.RERANKER_ENABLED, 
                                    help="Use a reranker to improve retrieval results")
            use_hybrid_search = st.checkbox("Use Hybrid Search", value=config.USE_HYBRID_SEARCH, 
                                         help="Combine semantic search with keyword search")
            
            vector_weight = config.VECTOR_WEIGHT
            if use_hybrid_search:
                vector_weight = st.slider("Vector Search Weight", min_value=0.0, max_value=1.0, 
                                       value=config.VECTOR_WEIGHT, step=0.1,
                                       help="Weight given to semantic search results (vs keyword search)")
            
            # Response validation options
            check_for_hallucinations = st.checkbox("Check for Hallucinations", value=config.HALLUCINATION_CHECK_ENABLED, 
                                               help="Check generated responses for hallucinations")
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, 
                                          value=0.6, step=0.1,
                                          help="Minimum confidence score to accept answers (below this will trigger fallbacks)")
            
            # LLM options
            st_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, 
                                     value=config.DEFAULT_LLM_TEMPERATURE, step=0.1, 
                                     help="Higher temperature means more creative but potentially less accurate responses")
            
            # Number of documents to retrieve
            retrieval_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, 
                                  value=config.RETRIEVAL_K, help="Number of documents to retrieve from the vector store")
            
            # Feedback mechanism
            st.subheader("Feedback")
            feedback = st.radio("How satisfied are you with the responses?", 
                             options=["", "Very Satisfied", "Satisfied", "Neutral", "Unsatisfied", "Very Unsatisfied"])
            
            if feedback and self.metrics_manager:
                satisfaction_mapping = {
                    "Very Satisfied": 1.0,
                    "Satisfied": 0.75,
                    "Neutral": 0.5,
                    "Unsatisfied": 0.25,
                    "Very Unsatisfied": 0.0
                }
                self.metrics_manager.set_user_satisfaction(satisfaction_mapping[feedback])
                st.success("Thank you for your feedback!")
            
            # Return the selected options
            return {
                "use_reranker": use_reranker,
                "use_hybrid_search": use_hybrid_search,
                "vector_weight": vector_weight,
                "check_for_hallucinations": check_for_hallucinations,
                "confidence_threshold": confidence_threshold,
                "temperature": st_temperature,
                "retrieval_k": retrieval_k
            }
    
    def render_chat_history(self):
        """Render the conversation history."""
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
                self.render_retrieved_sources(message.get("retrieved_docs", []), message.get("retrieval_settings", {}))
                
                # Show hallucination warning if needed
                if "hallucination_check" in message and message["hallucination_check"]["is_hallucination"]:
                    with st.chat_message("assistant").container():
                        st.warning("⚠️ This response may contain information not found in the document")
                        
                # Show validation details if available
                if "hallucination_check" in message:
                    with st.expander("Validation Details"):
                        st.write(message["hallucination_check"])
                        
                        # Show unverified claims if available
                        if "unverified_claims" in message["hallucination_check"] and message["hallucination_check"]["unverified_claims"]:
                            st.subheader("Unverified Claims")
                            for claim in message["hallucination_check"]["unverified_claims"]:
                                st.error(f"• {claim}")
                                
                        # Show verified claims if available
                        if "verified_claims" in message["hallucination_check"] and message["hallucination_check"]["verified_claims"]:
                            st.subheader("Verified Claims")
                            for claim in message["hallucination_check"]["verified_claims"]:
                                st.success(f"• {claim}")

    def render_retrieved_sources(self, retrieved_docs, retrieval_settings=None):
        """Render retrieved chunks so users can inspect the grounding context."""
        if not retrieved_docs:
            return

        with st.expander("Retrieved Sources"):
            if retrieval_settings:
                st.caption(
                    "k={retrieval_k} | hybrid={use_hybrid_search} | reranker={use_reranker} | "
                    "vector_weight={vector_weight}".format(**retrieval_settings)
                )

            for doc in retrieved_docs:
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "unknown source")
                page = metadata.get("page")
                page_label = f", page {page}" if page is not None else ""
                st.markdown(f"**Source {doc.get('rank', '?')}**: `{source}`{page_label}")
                st.write(doc.get("preview") or doc.get("content", "")[:300])
    
    def process_query(self, query: str, options: Dict[str, Any]):
        """
        Process a user query and update the UI.
        
        Args:
            query: The user's question
            options: Options from the sidebar
        """
        if not query.strip():
            st.warning("Please enter a valid question.")
            return
            
        # Display user message
        st.chat_message("user").write(query)
        st.session_state.conversation_history.append({"role": "user", "content": query})
        
        try:
            # Show processing status
            with st.status("Retrieving relevant information..."):
                # Call RAG service to process query
                result = self.rag_service.query(query, **options)
                
            # Display the response
            response_message = st.chat_message("assistant")
            response_message.write(result["response"])
            with response_message.container():
                self.render_retrieved_sources(
                    result.get("retrieved_docs", []),
                    result.get("retrieval_settings", {})
                )
            
            # Show hallucination warning if needed
            if "hallucination_check" in result and result["hallucination_check"]["is_hallucination"]:
                with response_message.container():
                    st.warning("⚠️ This response may contain information not found in the document")
                    with st.expander("Validation Details"):
                        st.write(result["hallucination_check"]["reasoning"])
                        
                        # Show unverified claims if available
                        if result["hallucination_check"]["unverified_claims"]:
                            st.subheader("Unverified Claims")
                            for claim in result["hallucination_check"]["unverified_claims"]:
                                st.error(f"• {claim}")
                                
                        # Show verified claims if available
                        if result["hallucination_check"]["verified_claims"]:
                            st.subheader("Verified Claims")
                            for claim in result["hallucination_check"]["verified_claims"]:
                                st.success(f"• {claim}")
            
            # Store in conversation history
            history_entry = {
                "role": "assistant", 
                "content": result["response"], 
                "validation_info": result.get("validation_info", {}),
                "retrieved_docs": result.get("retrieved_docs", []),
                "retrieval_settings": result.get("retrieval_settings", {})
            }
            
            if "hallucination_check" in result:
                history_entry["hallucination_check"] = result["hallucination_check"]
                
                # Update metrics
                if self.metrics_manager:
                    self.metrics_manager.record_hallucination_score(
                        result["hallucination_check"]["confidence_score"]
                    )
                    
            st.session_state.conversation_history.append(history_entry)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"Error processing query: {str(e)}")
    
    def render(self):
        """Render the complete chat UI."""
        self.render_header()
        options = self.render_sidebar()
        self.render_chat_history()
        
        # Input area
        query = st.chat_input('Ask a question about the PDF')
        
        if query:
            self.process_query(query, options) 
