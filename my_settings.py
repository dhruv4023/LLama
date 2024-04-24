import logging
import sys
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI


class LLMSettings():
    def __init__(
        self,
        model_name,
        query_wrapper_prompt=None,
        max_new_tokens=256,
        context_window=2048,
        device_map="auto",
        temperature=0.25,
        do_sample=False,
        max_length=2048,
        torch_dtype=torch.float32,
    ):
        self.model_name = model_name
        self.query_wrapper_prompt = query_wrapper_prompt
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.device_map = device_map
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_length = max_length
        self.torch_dtype = torch_dtype

        # Load environment variables
        self._load_env_variables()

        # Authenticate with Hugging Face
        self._authenticate_with_huggingface()

    def _configure_mongodb(self):
        self.MONGO_DB_URI = os.getenv("MONGO_DB_URI")

    def _configure_logging(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    def _load_env_variables(self):
        load_dotenv()

    def _authenticate_with_huggingface(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token:
            login(token=token, write_permission=True, add_to_git_credential=True)

    def _set_huggingface_embedding(self):
        Settings.embed_model = HuggingFaceEmbedding()

    def _setup_huggingface_llm_inference_api(self):
        from llama_index.core import PromptTemplate

        self.llm = HuggingFaceInferenceAPI(
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            generate_kwargs={
                "temperature": self.temperature,
                "do_sample": self.do_sample,
            },
            query_wrapper_prompt=PromptTemplate(self.query_wrapper_prompt),
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            device_map=self.device_map,
            tokenizer_kwargs={"max_length": self.max_length},
            model_kwargs={"torch_dtype": self.torch_dtype},
        )

        # Set chunk size
        Settings.chunk_size = 512
        Settings.llm = self.llm


# Example usage:
# model = MyHuggingFaceModel(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")
