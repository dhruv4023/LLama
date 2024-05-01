MIXTRAL_TEXT_GENERATION_MODEL_REPO = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# TEXT_GENERATION_MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct"
# TEXT_GENERATION_MODEL_REPO = "meta-llama/Meta-Llama-3-8B"
# TEXT_GENERATION_MODEL_REPO = "thenlper/gte-small"
# TEXT_GENERATION_MODEL_REPO = "openai/whisper-large-v3"
# TEXT_GENERATION_MODEL_REPO = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
# TEXT_GENERATION_MODEL_REPO = "StabilityAI/stablelm-tuned-alpha-3b"
# TEXT_GENERATION_MODEL_REPO = "Writer/camel-5b-hf"


import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import torch

torch.cuda.is_available()
print("DEVICE:", "GPU" if torch.cuda.is_available() else "CPU")


from llama_index.core import Settings
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import (
    HuggingFaceInferenceAPIEmbedding,
    HuggingFaceEmbedding,
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM


Settings.embed_model = HuggingFaceEmbedding(cache_folder="./tmp/models/")

Settings.llm = HuggingFaceInferenceAPI(
    tokenizer_name=MIXTRAL_TEXT_GENERATION_MODEL_REPO,
    model_name=MIXTRAL_TEXT_GENERATION_MODEL_REPO,
    device_map="auto",
)


# Settings.llm = HuggingFaceLLM(
#     model_name=TEXT_GENERATION_MODEL_REPO,
#     tokenizer=AutoTokenizer.from_pretrained(
#         TEXT_GENERATION_MODEL_REPO,
#         padding_side="left",
#     ),
#     tokenizer_name=TEXT_GENERATION_MODEL_REPO,
#     device_map="auto", # "cuda:0"
#     context_window=512,
#     tokenizer_kwargs={"max_length": 2048},
# #     model_kwargs={"torch_dtype": torch.float16},
# # )

# Settings.llm = HuggingFaceLLM(
#     context_window=2048,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.25, "do_sample": False},
#     tokenizer_name="Writer/camel-5b-hf",
#     model_name="Writer/camel-5b-hf",
#     device_map="auto",
#     tokenizer_kwargs={"max_length": 2048},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )
