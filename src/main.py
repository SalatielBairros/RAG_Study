import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain.tools import tool
from typing import Any
from pdf_rag_service import PdfRagService
from pdf_rag_chat import PdfRagChatAgent
from rich.console import Console
from langchain_ollama.chat_models import ChatOllama

console = Console()

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

pdf_rag_service = PdfRagService(pdf_directory="data/salatiel_classes/")

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

if __name__ == "__main__":

    with console.status("[bold green]Criando o sistema de recuperação de informações...", spinner="dots") as status:
        ensemble_retriever = pdf_rag_service.create_retriever()

    # ollama_model = ChatOllama(model="gemma4:e4b")
    # ollama_model = ChatOllama(model="granite4.1:8b-q6_K")
    # ollama_model = ChatOllama(model="qwen3.5:latest")
    # ollama_model = ChatOllama(model="ministral-3:14b")
    # ollama_model = ChatOllama(model="llama3.1:8b-instruct-q8_0")

    pdf_rag_chat_agent = PdfRagChatAgent(
        system_initial_prompt=system_prompt,
        temperature=0.0,
        rag_tools=[retrieve_context, get_filenames],
        verbose=False,
        # model=ollama_model
    )

    pdf_rag_chat_agent.run_console_chat()