# MIXTRAL_TEXT_GENERATION_MODEL_REPO = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# TEXT_GENERATION_MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct"
# TEXT_GENERATION_MODEL_REPO = "meta-llama/Meta-Llama-3-8B"
# TEXT_GENERATION_MODEL_REPO = "nvidia/Llama3-ChatQA-1.5-8B"
# TEXT_GENERATION_MODEL_REPO = "openai/whisper-large-v3"
# TEXT_GENERATION_MODEL_REPO = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
# TEXT_GENERATION_MODEL_REPO = "StabilityAI/stablelm-tuned-alpha-3b"
# TEXT_GENERATION_MODEL_REPO = "HuggingFaceH4/zephyr-7b-alpha"
# EMBEDDING_MODEL = "text-multilingual-embedding-preview-0409"
# TEXT_GENERATION_MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct"
# TEXT_GENERATION_MODEL_REPO = "openai-community/gpt2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
# EMBEDDING_MODEL =  "thenlper/gte-small"
# TEXT_GENERATION_MODEL_REPO = "thenlper/gte-small"
TEXT_GENERATION_MODEL_REPO = "mistralai/Mistral-7B-v0.1"


import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import torch

torch.cuda.is_available()
print("DEVICE:", "GPU" if torch.cuda.is_available() else "CPU")


from llama_index.core import Settings
from llama_index.embeddings.huggingface import (
    HuggingFaceInferenceAPIEmbedding,
    HuggingFaceEmbedding,
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter


Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

#### Inference API
Settings.llm = HuggingFaceInferenceAPI(
    tokenizer_name=TEXT_GENERATION_MODEL_REPO,
    model_name=TEXT_GENERATION_MODEL_REPO,
    device_map="cuda:0",
)

# from llama_index.llms.gemini import Gemini

# Settings.llm = Gemini(temperature=0.82, max_tokens=10000)


# ### Local LLM
# from transformers import pipeline

# generator = pipeline("text-generation", model="gpt2")
# Settings.llm = HuggingFaceLLM(
#     model_name=TEXT_GENERATION_MODEL_REPO,
#     tokenizer=AutoTokenizer.from_pretrained(
#         TEXT_GENERATION_MODEL_REPO,
#         padding_side="left",
#     ),
#     tokenizer_name=TEXT_GENERATION_MODEL_REPO,
#     device_map="cuda:0",  # "cuda:0"
#     context_window=512,
#     tokenizer_kwargs={"max_length": 1024},
#     model_kwargs={
#         "torch_dtype": torch.float16,
#         # "pad_token_id": generator.tokenizer.eos_token_id,
#     },
# )

# Settings.llm = HuggingFaceLLM(
#     context_window=512,
#     max_new_tokens=1024,
#     generate_kwargs={"temperature": 0.25, "do_sample": False},
#     tokenizer_name=TEXT_GENERATION_MODEL_REPO,
#     model_name=TEXT_GENERATION_MODEL_REPO,
#     device_map="cuda",
#     tokenizer_kwargs={"max_length": 2048},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )
