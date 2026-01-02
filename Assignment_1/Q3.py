from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text_list = ["The movie was absolutely fantastic, with brilliant acting and a gripping storyline that kept me engaged throughout.",
    "I expected a lot more from this film, but the plot was slow and the characters felt shallow.",
    "The visuals were stunning, but the story itself was average and somewhat predictable.",
    "This was a complete waste of time; the script was terrible and the performances were painful to watch.",
    "A heartfelt and beautifully made film that left me feeling inspired and emotional by the end."]

out = classifier(text_list)
for review, result in zip(text_list, out):
    print(f"Review: {review}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.3f}\n")
