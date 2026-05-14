from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages.tool import ToolMessage
from rich.console import Console
from rich.markdown import Markdown

class PdfRagChatAgent:
    def __init__(self, system_initial_prompt: str, temperature: float = 0.1, rag_tools: list = [], model: BaseChatModel | None = None, verbose: bool = False):
        self.system_initial_prompt = system_initial_prompt
        self.temperature = temperature
        self.rag_tools = rag_tools
        self.model = model
        self.verbose = verbose
        self.console = Console()

        if(model is None):
            self.model = ChatGoogleGenerativeAI(
                model="models/gemini-3.1-flash-lite",
                temperature=temperature
            )

        self.agent = self.__create_agent__()

    def __create_agent__(self):
        if(self.model is None):
           raise ValueError("Model is not initialized")
        return create_agent(model=self.model, tools=self.rag_tools, system_prompt=self.system_initial_prompt, checkpointer=MemorySaver())
    
    def ask(self, query: str):
        message = {}
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            message = event["messages"][-1]
            if(type(message) is not ToolMessage and self.verbose):
                message.pretty_print()

            print("\n---\n")
        
        if(type(message) is AIMessage):
            print("================================== Final response ==================================")
            for c in message.content:
                if(isinstance(c, dict) and 'text' in c):
                    print(c['text'])

    def run_console_chat(self):
        self.console.print("\n\n[bold blue]SISTEMA DE PESQUISA ATUALIZADO (v2026)[/bold blue]")
        self.console.print("Digite 'sair' para encerrar.\n")

        config = RunnableConfig(
            configurable={"thread_id": "usuario_salatiel_ebd"}
        )

        while True:
            user_input = input("\nVocê: ")
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                break

            final_message = None
            with self.console.status("[bold green]O agente está pensando...", spinner="dots") as status:
                for event in self.agent.stream(
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
                
                self.console.print("\n[bold magenta]Assistente:[/bold magenta]")
                self.console.print(md)
                self.console.print("[dim]" + "—" * self.console.width + "[/dim]\n")