# ragas_integration.py --> RAGAS evaluation integration for Streamlit

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.llms import LangchainLLMWrapper  
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    st.sidebar.warning("Install RAGAS: pip install ragas datasets")

from config import GROQ_API_KEY

class RAGASEvaluator:
    """RAGAS evaluator for real-time quality assessment"""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.wrapped_llm = None
        self.wrapped_embeddings = None
        self.metrics = []
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAGAS components"""
        if not RAGAS_AVAILABLE:
            return
            
        try:
            # Initialize LLM for evaluation (using a fast model)
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",  # Fast model for evaluation
                groq_api_key=GROQ_API_KEY,
                temperature=0.0,
                max_tokens=1000,
                n = 1
            )
            
            # Initialize embeddings  
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Wrap for RAGAS
            self.wrapped_llm = LangchainLLMWrapper(self.llm)
            self.wrapped_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
            
            # Initialize metrics (only those that don't require ground truth)
            self.metrics = [
                Faithfulness(llm=self.wrapped_llm),
                # AnswerRelevancy(llm=self.wrapped_llm, embeddings=self.wrapped_embeddings)
            ]
            
        except Exception as e:
            st.error(f"Failed to initialize RAGAS components: {e}")
            self.metrics = []
    
    def is_available(self) -> bool:
        """Check if RAGAS evaluation is available"""
        return RAGAS_AVAILABLE and len(self.metrics) > 0
    
    def evaluate_response(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Evaluate a single response and return metrics"""
        if not self.is_available():
            return {}
            
        try:
            # Create dataset for single response
            eval_data = [{
                "question": question,
                "answer": answer, 
                "contexts": contexts
            }]
            
            dataset = Dataset.from_list(eval_data)
            
            # Run evaluation
            results = evaluate(dataset=dataset, metrics=self.metrics)
            
            # Extract scores
            df = results.to_pandas()
            scores = {}
            
            if len(df) > 0:
                row = df.iloc[0]
                if 'faithfulness' in df.columns and pd.notna(row['faithfulness']):
                    scores['faithfulness'] = float(row['faithfulness'])
                # if 'answer_relevancy' in df.columns and pd.notna(row['answer_relevancy']):
                #     scores['answer_relevancy'] = float(row['answer_relevancy'])
            
            return scores
            
        except Exception as e:
            st.error(f"RAGAS evaluation failed: {e}")
            return {}

# Global evaluator instance
@st.cache_resource
def get_ragas_evaluator():
    """Get cached RAGAS evaluator instance"""
    return RAGASEvaluator()

def display_ragas_metrics(scores: Dict[str, float]):
    """Display RAGAS metrics in Streamlit UI"""
    if not scores:
        return
        
    st.markdown("### 📊 Response Quality Metrics")
    
    # Adjust columns based on available metrics
    if len(scores) == 2:
        col1, col2 = st.columns(2)
        cols = [col1, col2]
    else:
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
    
    col_idx = 0
    
    with cols[col_idx]:
        if 'faithfulness' in scores:
            score = scores['faithfulness']
            st.metric(
                label="🎯 Faithfulness",
                value=f"{score:.3f}",
                help="How well the answer is grounded in retrieved contexts (0-1)"
            )
            st.progress(score)
            col_idx += 1
    
    # if col_idx < len(cols):
    #     with cols[col_idx]:
    #         if 'answer_relevancy' in scores:
    #             score = scores['answer_relevancy'] 
    #             st.metric(
    #                 label="🔍 Answer Relevancy", 
    #                 value=f"{score:.3f}",
    #                 help="How relevant the answer is to the question (0-1)"
    #             )
    #             st.progress(score)
    #             col_idx += 1
    
    # if col_idx < len(cols):
    #     with cols[col_idx]:
    #         if 'context_precision' in scores:
    #             score = scores['context_precision']
    #             st.metric(
    #                 label="📋 Context Precision",
    #                 value=f"{score:.3f}",
    #                 help="Precision of retrieved context relevance (0-1)"
    #             )
    #             st.progress(score)
    
    # Overall quality indicator
    if len(scores) > 0:
        avg_score = sum(scores.values()) / len(scores)
        quality_level = get_quality_level(avg_score)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: {get_quality_color(avg_score)};">
            <strong>Overall Quality: {quality_level} ({avg_score:.3f})</strong>
        </div>
        """, unsafe_allow_html=True)

def get_score_color(score: float) -> str:
    """Get color based on score"""
    if score >= 0.8:
        return "#28a745"  # Green
    elif score >= 0.6:
        return "#ffc107"  # Yellow  
    else:
        return "#dc3545"  # Red

def get_quality_level(score: float) -> str:
    """Get quality level description"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Needs Improvement"

def get_quality_color(score: float) -> str:
    """Get background color for quality indicator"""
    if score >= 0.8:
        return "#d4edda"  # Light green
    elif score >= 0.6:
        return "#fff3cd"  # Light yellow
    else:
        return "#f8d7da"  # Light red

def show_ragas_settings():
    """Show RAGAS settings in sidebar"""
    if not RAGAS_AVAILABLE:
        st.sidebar.warning("⚠️ RAGAS not available. Install with: pip install ragas")
        return False
        
    st.sidebar.markdown("### 📊 Quality Metrics")
    
    enable_ragas = st.sidebar.checkbox(
        "Enable Response Evaluation", 
        value=False,
        help="Evaluate response quality using RAGAS metrics"
    )
    
    if enable_ragas:
        st.sidebar.info("""
        **Metrics Included:**
        - **Faithfulness**: Answer grounding in context
        """)
        
        # - **Answer Relevancy**: Question-answer alignment
        # ⏱️ *Note: Evaluation adds ~5-10 seconds per response*
        # 📝 *Context Precision requires ground truth data and is not available in real-time mode*
    
    return enable_ragas