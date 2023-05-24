import argparse
import os
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import AgentType
from langchain.agents import initialize_agent

from langchain import PromptTemplate, OpenAI, LLMChain

from langchain.tools import BaseTool, StructuredTool, Tool, tool

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores import FAISS
from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter

load_dotenv()

# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
llm = AzureChatOpenAI(
    openai_api_base=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version="2023-05-15"
)

def build_prompt():
    template="너는 나의 코드 분석을 도와주는 역할을 수행해. 너는 tool들을 활용해서 코드를 검색하고, 해당 결과를 활용해 나의 질문에 답하도록 노력해."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # prompt = PromptTemplate(
    #     template="너는 나의 코드 분석을 도와주는 역할을 수행해. 너는 tool들을 활용해서 코드를 검색하고, 해당 결과를 활용해 나의 질문에 답하도록 노력해.",
    #     # input_variables=["input_language", "output_language"],
    # )
    # system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    # chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages()

    return chat_prompt

from langchain.document_loaders import TextLoader
class DB:
    def __init__(self, repo_path) -> None:
        self.repo_path = repo_path
        self.load_docs()
        self.load_retriever()
        
    def load_docs(self):
        # loader = DirectoryLoader('', glob="*.py", loader_cls=TextLoader)
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            for file in filenames:
                if file.endswith('.py') and '/.venv/' not in dirpath:
                    try: 
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e: 
                        pass
        print(f'{len(self.docs)}')

    def load_retriever(self):
        # documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(self.docs)
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("AZURE_API_BASE"),
            deployment=os.environ.get("EMBEDDING_SKILL_DEPLOYMENT_NAME"),
            chunk_size=1
        )
        self.db = FAISS.from_documents(documents=docs, embedding=embeddings)

        self.retriever = self.db.as_retriever()

chat_history = []
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

chat_prompt = build_prompt()


# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# agent_chain = initialize_agent(
#     tools=[],
#     llm=llm, 
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory
# )



@tool
def serach_code(query: str) -> str:
    """Find codes and return."""
    docs = db.similarity_search(str)
    return f"Code chunks for code analysis and query response are as follows.\n{docs}"

def generate_response(qa, prompt):
    """
    Generate a response using specified prompt.
    """
    # messages = chat_prompt.format_prompt(text=prompt)
    # # messages = chat_prompt.format_prompt(text=prompt).to_messages()
    # print(messages)
    # response = agent_chain.run(input=messages)
    # return response

    result = qa({"question": prompt, "chat_history": chat_history})
    chat_history.append((prompt, result["answer"]))
    return result["answer"]
# 

def get_text():
    """Create a Streamlit input field and return the user's input."""
    input_text = st.text_input("", key="input")
    return input_text


def search_db(db, query):
    """Search for a response to the query in the DeepLake database."""
    # Create a retriever from the DeepLake instance
    retriever = db.as_retriever()
    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    # Create a ChatOpenAI model instance
    model = ChatOpenAI(model="gpt-3.5-turbo")
    # Create a RetrievalQA instance from the model and retriever
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    # Return the result of the query
    return qa.run(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activeloop_dataset_path", type=str, required=True)
    args = parser.parse_args()

    run_chat_app(args.activeloop_dataset_path)