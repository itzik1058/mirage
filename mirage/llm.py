from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from mirage.config import CACHE_PATH, DATA_PATH

CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE
N_BATCH = 512
N_GPU_LAYERS = 100


def prompt_template(system_prompt: str):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = f"{B_SYS}{system_prompt}{E_SYS}"
    instruction = """
    Context: {history} \n {context}
    User: {question}
    """

    prompt_template = f"{B_INST}{SYSTEM_PROMPT}{instruction}{E_INST}"
    return PromptTemplate(
        input_variables=["history", "context", "question"],
        template=prompt_template,
    )


def agent(system_prompt: str):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        # model_kwargs={"device": "cuda"},
    )
    chroma = Chroma(
        persist_directory=str(DATA_PATH / "chroma"),
        embedding_function=embeddings,
    )
    retriever = chroma.as_retriever()
    prompt = prompt_template(system_prompt)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    model_path = hf_hub_download(
        repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
        filename="llama-2-7b-chat.Q4_K_M.gguf",
        resume_download=True,
        cache_dir=CACHE_PATH / "hf",
    )
    kwargs = {
        "model_path": model_path,
        "n_ctx": CONTEXT_WINDOW_SIZE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_batch": N_BATCH,
        # "n_gpu_layers": N_GPU_LAYERS,
    }
    llm = LlamaCpp(**kwargs)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # refine, map_reduce, map_rerank
        retriever=retriever,
        return_source_documents=True,  # verbose=True,
        callbacks=callback_manager,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
