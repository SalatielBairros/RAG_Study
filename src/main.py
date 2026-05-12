import os
import logging
import time
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig, ConfigurableField
from google import genai
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
import nltk
import ssl
from typing import Any
from langgraph.checkpoint.memory import MemorySaver 
from rich.console import Console
from rich.markdown import Markdown

console = Console()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
ensemble_retriever: EnsembleRetriever

def load_pdfs() -> list[Document]:
    start_time = time.time()
    loader = PyPDFDirectoryLoader("data/salatiel_classes/")
    documents = loader.load()
    execution_time = time.time() - start_time
    logging.info(f"Loaded {len(documents)} documents from PDFs. Execution time: {execution_time:.2f}s")
    return documents

def split_documents_texts(documents: list[Document]) -> list[Document]:
    start_time = time.time()
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    execution_time = time.time() - start_time
    logging.info(f"Splited documents into {len(split_docs)} chunks. Execution time: {execution_time:.2f}s")
    return split_docs

def create_vector_store() -> tuple[Chroma, list[Document]]:
    start_time = time.time()
    persist_directory="data/db/chroma"
    collection_name="salatiel_classes"
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    documents = load_pdfs()
    splited_docs = split_documents_texts(documents)

    if(os.path.exists(persist_directory) and os.listdir(persist_directory)):
        logging.info("Persisted vector store found. Loading existing vector store...")
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings_model
        )
        execution_time = time.time() - start_time
        logging.info(f"Loaded existing vector store. Execution time: {execution_time:.2f}s")
        return vectorstore, splited_docs

    logging.info("No persisted vector store found. Creating new vector store...")
    vectorstore = Chroma.from_documents(
        documents=splited_docs,
        embedding=embeddings_model,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    execution_time = time.time() - start_time
    logging.info(f"Created vector store. Execution time: {execution_time:.2f}s")
    return vectorstore, splited_docs

def create_retriever() -> EnsembleRetriever:
    start_time = time.time()
    vectorstore, split_docs = create_vector_store()

    bm25_retriever = BM25Retriever.from_documents(
        split_docs,
        preprocess_func=word_tokenize)
    bm25_retriever.k = 10

    vectorstore_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20}
    ).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
            name="Search Kwargs",
            description="The search kwargs to use"
        )
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever], 
        weights=[0.4, 0.6])
    
    execution_time = time.time() - start_time
    logging.info(f"Created ensemble retriever. Execution time: {execution_time:.2f}s")
    return ensemble_retriever

def get_avaliable_chat_models():
    client = genai.Client()
    models = client.models.list()
    logging.info("Available chat models:")
    for model in models:
        logging.info(f"- {model.name} (Version: {model.version} | Description: {model.description} | Actions: {model.supported_actions})")

@tool("retrieve_context", description="Recupera contexto sobre manuscritos das aulas da Escola Bíblica da Igreja Batista Conde.", response_format="content_and_artifact")
def retrieve_context(query: str, filename: str) -> tuple[str, list[Document]]:
    search_kwargs: dict[str, Any] = {"k": 10, "fetch_k": 20}

    if(filename):
        search_kwargs["filter"] = {"source": f"data/salatiel_classes/{filename}"}

    retrieved_docs = ensemble_retriever.with_config(configurable={"search_kwargs": search_kwargs}).invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    logging.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query}' with filename filter: '{filename}'")
    return serialized, retrieved_docs

@tool("get_filenames", description="Busca o nome dos arquivos dos manuscritos das aulas. Isso serve para ajudar a identificar o assunto do qual cada arquivo trata.")
def get_filenames() -> list[str]:
    directory = "data/salatiel_classes/"
    return [filename for filename in os.listdir(directory) if filename.endswith(".pdf")]

def agent(local_model: bool = True, temperature: float = 0.1):
    system_prompt = (
        "Você é um assistente de pesquisa especializado em fornecer informações relevantes com base em documentos de manuscritos de aulas da Escola Bíblica da Igreja Batista Conde. "
        "A resposta deve ser formatada como Markdown, utilizando os recursos de formatação disponíveis para organizar as informações de maneira clara e legível."
        "Não use símbolos de formatação, como negrito ou itálico, pois eles podem não ser renderizados corretamente."
        "Se precisar recuperar informações, use a ferramenta 'retrieve_context' para obter os documentos relevantes."
        "Caso precise de mais detalhes de um arquivo específico, é possível usar a ferramenta 'get_filenames' para obter os nomes dos arquivos disponíveis e, em seguida, usar 'retrieve_context' com o nome do arquivo para recuperar informações específicas. "
        "Utilize apenas as informações dos documentos recuperados para formular sua resposta, e cite as fontes quando possível, incluindo a página."
        "Faça tantas chamadas às ferramentas quantas forem necessárias para obter as informações relevantes antes de formular sua resposta final."
        "Busque sempre fazer pelo menos duas chamadas à ferramentas. Formule novas queries e perguntas para fazer aos documentos para enriquecer ao máximo a resposta."
        "Caso alguma tool ou parâmetro tenha feito falta para obter mais dados, acrescente ao final da resposta uma seção chamada 'Limitações' e explique o que faltou para obter uma resposta mais completa."
    )
    tools = [retrieve_context, get_filenames]

    if(local_model):
        llm = ChatOllama(
            model="gemma4:e4b",
            temperature=temperature,
            reasoning=True
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=temperature
        )

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=MemorySaver())

def ask(agent, query: str, verbose: bool = False):
    message = {}
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values"
    ):
        message = event["messages"][-1]
        if(type(message) is not ToolMessage and verbose):
            message.pretty_print()

        print("\n---\n")
    
    if(type(message) is AIMessage):
        print("================================== Final response ==================================")
        for c in message.content:
            if(isinstance(c, dict) and 'text' in c):
                print(c['text'])

def run_chat():
    console.print("\n\n[bold blue]SISTEMA DE PESQUISA ATUALIZADO (v2026)[/bold blue]")
    console.print("Digite 'sair' para encerrar.\n")

    config = RunnableConfig(
        configurable={"thread_id": "usuario_salatiel_ebd"}
    )

    research_agent = agent()
    while True:
        user_input = input("\nVocê: ")
        
        if user_input.lower() in ["sair", "exit", "quit"]:
            break

        final_message = None
        with console.status("[bold green]O agente está pensando...", spinner="dots") as status:
            for event in research_agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            ):
                final_message = event["messages"][-1]
                if isinstance(final_message, ToolMessage):
                    status.update("[bold cyan]Buscando informações nos manuscritos...")
                if isinstance(final_message, AIMessage):
                    status.update("[bold green]O agente está pensando...")


        if isinstance(final_message, AIMessage):
            content = final_message.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                    elif isinstance(block, str):
                        text_parts.append(block)
                text = "\n".join(text_parts)
            else:
                text = str(content)

            md = Markdown(text)
            
            console.print("\n[bold magenta]Assistente:[/bold magenta]")
            console.print(md)
            console.print("[dim]" + "—" * console.width + "[/dim]\n")


if __name__ == "__main__":
    # get_avaliable_chat_models()

    with console.status("[bold green]Criando o sistema de recuperação de informações...", spinner="dots") as status:
        ensemble_retriever = create_retriever()

    # response = get_filenames()
    # print(response)
    # research_agent = agent()
    # ask(research_agent, "Quais são os principais temas abordados nas aulas do Salatiel? Cite as fontes dos documentos recuperados.", verbose=True)
    run_chat()
    # _, retrieved_docs = retrieve_context("qual é o tema principal deste documento?", "Idolatria.pdf")
    # print(retrieved_docs)