from pathlib import Path
import yaml
from typing import List, Dict, Any, Union
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
import torch

class PretrainedRecommender:

    def __init__(self, documents: List[Document], config: Union[Dict[str, Any], Path] = None):

        self.documents = documents
        default_config_path = Path(__file__).parent.parent / "config" / "pretrained_config.yaml"
        self.config = self._load_config(config if config is not None else default_config_path)
        self._validate_config()
        self._validate_documents()

        # Инициализация компонентов
        self.embeddings = self._load_embeddings()
        self.vector_store = self._load_vector_store()
        self.tokenizer, self.model = self._load_base_model()
        self.llm = self._create_pipeline()
        self.prompt = self._setup_prompt()
        self.qa_chain = self._setup_retrieval_chain()

    def _load_config(self, config: Union[Dict[str, Any], Path]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _validate_config(self) -> None:
        required_keys = ["embedding_model", "llm", "retriever", "paths"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Key '{key}' not found in config")
        if "name" not in self.config["embedding_model"] or "device" not in self.config["embedding_model"]:
            raise KeyError("Invalid 'embedding_model' configuration")
        if "model_name" not in self.config["llm"] or "params" not in self.config["llm"]:
            raise KeyError("Invalid 'llm' configuration")
        if "dataset" not in self.config["paths"]:
            raise KeyError("Invalid 'paths' configuration")

    def _validate_documents(self) -> None:
        if not self.documents or not isinstance(self.documents, list):
            raise ValueError("Documents must be a non-empty list")
        if not all(isinstance(doc, Document) for doc in self.documents):
            raise ValueError("All documents must be instances of langchain.docstore.document.Document")

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"]["name"],
            model_kwargs={"device": self.config["embedding_model"]["device"]}
        )

    def _load_vector_store(self) -> FAISS:
        faiss_index_path = Path(self.config["paths"]["faiss_index"])
        if faiss_index_path.exists():
            return FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
        vector_store = FAISS.from_documents(self.documents, self.embeddings)
        vector_store.save_local(faiss_index_path)
        return vector_store

    def _load_base_model(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.config["llm"]["quantization"]["load_in_4bit"],
            bnb_4bit_use_double_quant=self.config["llm"]["quantization"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=self.config["llm"]["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, self.config["llm"]["quantization"]["bnb_4bit_compute_dtype"])
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config["llm"]["model_name"])
        model = AutoModelForCausalLM.from_pretrained(
            self.config["llm"]["model_name"],
            quantization_config=quantization_config,
            device_map=self.config["llm"]["device_map"],
            torch_dtype=torch.float16
        )
        # Поддержка LoRA (для совместимости с fine_tuned_config.yaml)
        if "lora_path" in self.config["llm"]:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.config["llm"]["lora_path"])
        return tokenizer, model

    def _create_pipeline(self) -> HuggingFacePipeline:
        llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config["llm"]["params"]["max_new_tokens"],
            temperature=self.config["llm"]["params"]["temperature"],
            device_map=self.config["llm"]["device_map"]
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)

    def _setup_prompt(self) -> PromptTemplate:
        prompt_template = """
        You are a movie and TV series recommender expert. Use the provided context from the dataset to recommend films and series on any theme, focusing on genres, plot similarities, and user preferences. If the context lacks sufficient relevant matches, you may suggest additional films or series from your knowledge that fit the query, but prioritize the context for accuracy. Do not include documentaries, comedies, or unrelated genres unless highly relevant to the query. Do not repeat the context or question in the response.

        Context: {context}
        Question: {question}

        Answer format:
        - List up to 5 recommendations.
        - For each: Title, Year, Genres, Rating, Why it matches the query.
        - Be concise and accurate.
        Answer:
        """
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    def _setup_retrieval_chain(self) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs=self.config["retriever"]),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def recommend(self, query: str) -> Dict[str, Any]:
        result = self.qa_chain({"query": query})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }