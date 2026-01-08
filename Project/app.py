# app.py
import streamlit as st

from pipeline.controller import PipelineController
from a2a.conversation_log import ConversationLog

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Collaborative Paper Finder",
    layout="wide"
)

st.title("📚 Collaborative Paper Finder – Phase 1")

# -------------------------
# User Input
# -------------------------
query = st.text_input("Enter research topic")

# -------------------------
# Run Pipeline
# -------------------------
if st.button("Search") and query.strip():

    # Create a fresh conversation log for THIS run
    conversation_log = ConversationLog()

    # Create pipeline controller with logging enabled
    controller = PipelineController(conversation_log)

    try:
        with st.spinner("Running agents..."):
            papers = controller.run(query)

        # -------------------------
        # Main Results Area
        # -------------------------
        st.subheader("📄 Top Papers")

        for i, p in enumerate(papers, 1):
            st.markdown(f"### {i}. {p.title}")
            st.markdown(f"**Authors:** {', '.join(p.authors)}")
            st.markdown(f"**Year:** {p.year}")
            st.markdown(f"**Relevance Score:** `{p.relevance_score:.3f}`")
            st.markdown("**Abstract:**")
            st.markdown(p.abstract)
            st.markdown("---")

    except RuntimeError as e:
        st.error(str(e))

    # -------------------------
    # Sidebar: Agent Logs
    # -------------------------
    st.sidebar.title("🧠 Agent Execution Log")

    for step in conversation_log.entries:
        st.sidebar.markdown(step)
