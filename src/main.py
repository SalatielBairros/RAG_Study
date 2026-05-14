import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool
from typing import Any
from pdf_rag_service import PdfRagService
from pdf_rag_chat import PdfRagChatAgent
from rich.console import Console
from langchain_ollama.chat_models import ChatOllama
from gemini_service import GeminiService

console = Console()

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

ensemble_retriever: EnsembleRetriever

system_prompt = (
        "Você é um assistente de pesquisa especializado em fornecer informações relevantes com base em documentos de manuscritos de aulas da Escola Bíblica da Igreja Batista Conde."
        "O conteúdo desses manuscritos pode ser encontrado na tool retrieve_context, e o nome dos arquivos disponíveis pode ser obtido através da tool get_filenames."
        "A resposta deve ser formatada como Markdown, utilizando os recursos de formatação disponíveis para organizar as informações de maneira clara e legível."
        "Caso precise de mais detalhes de um arquivo específico, é possível usar a ferramenta 'get_filenames' para obter os nomes dos arquivos disponíveis e, em seguida, usar 'retrieve_context' com o nome do arquivo para recuperar informações específicas. "
        "Utilize apenas as informações dos documentos recuperados para formular sua resposta."
        "Cite os arquivos e suas respectivas páginas utilizadas na resposta."
        "Faça tantas chamadas às ferramentas quantas forem necessárias para obter as informações relevantes antes de formular sua resposta final."
        "Consulte tantos documentos quantos forem necessários para fornecer uma resposta completa e precisa. Ao final, revise a resposta e considere se buscar algum outro documento pode enriquecer a resposta antes de apresentá-la. "
        "Caso alguma tool ou parâmetro tenha feito falta para obter mais dados, acrescente ao final da resposta uma seção chamada 'Limitações' e explique o que faltou para obter uma resposta mais completa."
        "Seja preciso quanto ao conteúdo dos documentos, evitando generalizações e suposições. Se a resposta não puder ser formulada com base nas informações disponíveis, explique claramente o motivo e quais informações adicionais seriam necessárias para fornecer uma resposta completa. "
        "Não faça aplicações teológicas ou devocionais do assunto a não ser que seja solicitado explicitamente. Mantenha o foco em fornecer informações relevantes e precisas com base nos documentos disponíveis, e deixe as aplicações para o usuário."
    )


@tool("retrieve_context", description="Recupera contexto sobre manuscritos das aulas da Escola Bíblica da Igreja Batista Conde.", response_format="content_and_artifact")
def retrieve_context(query: str, filename: str) -> tuple[str, list[Document]]:
    search_kwargs: dict[str, Any] = {"k": 10, "fetch_k": 20}

    if(filename):
        search_kwargs["filter"] = {"source": f"data/salatiel_classes/{filename}"}

    global ensemble_retriever
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

def execute():
    # ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    # chunk_size = 2000
    # chunk_overlap = 200

    ollama_embeddings = OllamaEmbeddings(model="bge-m3:latest")
    chunk_size = 4000
    chunk_overlap = 400

    # ollama_embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    # chunk_size = 6000
    # chunk_overlap = 600

    pdf_rag_service = PdfRagService(
        pdf_directory="data/salatiel_classes/",
        embeddings_model=ollama_embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    with console.status("[bold green]Criando o sistema de recuperação de informações...", spinner="dots") as status:
        global ensemble_retriever
        ensemble_retriever = pdf_rag_service.create_retriever()

    # ollama_model = ChatOllama(model="gemma4:e4b")
    # ollama_model = ChatOllama(model="granite4.1:8b-q6_K")
    # ollama_model = ChatOllama(model="qwen3.5:latest")
    # ollama_model = ChatOllama(model="ministral-3:14b")
    # ollama_model = ChatOllama(model="llama3.1:8b-instruct-q8_0")

    # pdf_rag_chat_agent = PdfRagChatAgent(
    #     system_initial_prompt=system_prompt,
    #     temperature=0.0,
    #     rag_tools=[retrieve_context, get_filenames],
    #     verbose=False,
    #     model=ollama_model
    # )

    # pdf_rag_chat_agent.run_console_chat()

    if(pdf_rag_service.vectorstore is not None):
        search = pdf_rag_service.vectorstore.similarity_search("ofícios de Cristo", 5)
        console.print(search)



if __name__ == "__main__":
    execute()

    # GeminiService().get_avaliable_chat_models()