from schemas.paper import Paper
from schemas.summary import PaperSummary
from agents.comparison_agent import ComparisonAgent
from a2a.conversation_log import ConversationLog
from a2a.message_bus import MessageBus

# Dummy papers
papers = [
    Paper("BI-LSTM-CRF", [], "", 2015, "p1"),
    Paper("Highway LSTM", [], "", 2017, "p2"),
]

summaries = [
    PaperSummary(
        problem="Sequence tagging tasks",
        core_idea="BI-LSTM with CRF decoding",
        methodology="Bidirectional LSTM + CRF layer",
        key_results="Strong NLP performance",
        limitations="Task-specific",
        best_use_cases="NLP tagging"
    ),
    PaperSummary(
        problem="Language modeling",
        core_idea="Highway connections inside LSTM",
        methodology="Highway-enhanced LSTM",
        key_results="Improved ASR accuracy",
        limitations="Speech-focused",
        best_use_cases="ASR"
    ),
]

log = ConversationLog()
bus = MessageBus(log)

agent = ComparisonAgent("ComparisonAgent", bus)
comparison = agent.run(papers, summaries)

print(comparison)

print("\n--- Logs ---")
for msg in log.entries:
    print(msg)
