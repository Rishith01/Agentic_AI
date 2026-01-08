## WIDS_2025 — Work Summary


This repository serves as a centralized collection of coursework, experiments, and study notes from the WIDS_2025 workspace. It documents the implementation of various AI architectures, ranging from basic neural networks to advanced multi-agent systems.

📂 Repository Structure
🧠 Deep Learning & PyTorch
Files: Pytorch_basic/ (ANN, CNN, LSTM notebooks).

Work Done: Implemented Feed-Forward (ANN) and Convolutional (CNN) networks for Fashion-MNIST; built LSTM-based next-word predictors.

Key Concepts: Training loops, optimizers, hyperparameter tuning with Optuna, GPU acceleration, and sequence modeling.

🔍 RAG (Retrieval-Augmented Generation)
Files: RAG_from_scratch/ (Notebooks 1–18), chroma_db/.

Work Done: Built an end-to-end RAG pipeline including document chunking, embedding generation, and vector store persistence using ChromaDB.

Key Concepts: Vector embeddings, similarity search, indexing, and reducing LLM hallucinations through grounding.

🤖 Agentic Frameworks (LangChain & LangGraph)
Files: Hugging_Face_LLM_basics/, LangGraph/, Google ADK/.

Work Done: Created sequential and conditional graph pipelines, memory-enabled chatbots, and multi-agent coordination prototypes.

Key Concepts: Graph-based orchestration, tool-calling, agent-to-agent (A2A) communication, and multi-step reasoning patterns.

📝 Assignments & Problem Solving
Files: Assignment_1/, Assignment_2/.

Work Done: Practical algorithmic solutions and Python scripts for graded coursework.

Key Concepts: Algorithm design, efficient Python scripting, and modular code structure.

🛠️ How to Explore
Run Notebooks: Open any .ipynb file in the respective directory to view experiments and analysis.

Reproduce Training: Use the scripts in Pytorch_basic/ to re-run model training on the Fashion-MNIST dataset.

RAG Pipeline: Follow the sequential notebooks in RAG_from_scratch/ to understand the step-by-step construction of a retrieval system.