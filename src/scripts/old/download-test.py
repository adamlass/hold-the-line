from datasets import load_dataset

tedlium = load_dataset("LIUM/tedlium", "release1") # for Release 1

# see structure
print(tedlium)

# load audio sample on the fly
audio_input = tedlium["train"][0]["audio"]  # first decoded audio sample
transcription = tedlium["train"][0]["text"]  # first transcription

