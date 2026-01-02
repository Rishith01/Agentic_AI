from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
out = generator("In 2030, AI systems will",num_return_sequences = 2, max_new_tokens=50)
print("Generated output 1:\n", out[0]["generated_text"], "\n")
print("Generated output 2:\n", out[1]["generated_text"])