from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Artificial intelligence has rapidly evolved over the past decade, transforming the way people interact with technology in daily life. From recommendation systems on streaming platforms to advanced medical imaging tools, AI systems are now embedded in many critical applications. Businesses increasingly rely on machine learning models to analyze large volumes of data and make informed decisions faster than humans could. At the same time, concerns around data privacy, bias, and transparency have grown, prompting discussions about ethical AI development. As research continues, the focus is shifting toward building AI systems that are not only powerful but also reliable, fair, and aligned with human values."""

summary = summarizer(text, min_length=30, max_length=60)

print(f"Input length (chars): {len(text)}\n")
print(f"Summary: {summary[0]['summary_text']}")
print(f"Summary length (chars): {len(summary[0]['summary_text'])}")
