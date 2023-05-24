import argparse
import os
import sys
from dotenv import load_dotenv
from src.vector_db import process
from src.chat import generate_response, DB
from langchain.chains import ConversationalRetrievalChain
# Load environment variables from a .env file (containing OPENAI_API_KEY)
load_dotenv()
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import gradio as gr
import random
import time

import asyncio, httpx
import async_timeout

from typing import Optional, List
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
# def extract_repo_name(repo_url):
#     """Extract the repository name from the given repository URL."""
#     repo_name = repo_url.split("/")[-1].replace(".git", "")
#     return repo_name


# def process_repo(args):
#     """
#     Process the git repository by cloning it, filtering files, and
#     creating an Activeloop dataset with the contents.
#     """
#     repo_name = extract_repo_name(args.repo_url)
#     activeloop_username = os.environ.get("ACTIVELOOP_USERNAME")

#     if not args.activeloop_dataset_name:
#         args.activeloop_dataset_path = f"hub://{activeloop_username}/{repo_name}"
#     else:
#         args.activeloop_dataset_path = (
#             f"hub://{activeloop_username}/{args.activeloop_dataset_name}"
#         )

#     process(
#         args.repo_url,
#         args.include_file_extensions,
#         args.activeloop_dataset_path,
#         args.repo_destination,
#     )


# def chat(args):
#     """
#     Start the Streamlit chat application using the specified Activeloop dataset.
#     """
#     activeloop_username = os.environ.get("ACTIVELOOP_USERNAME")

#     args.activeloop_dataset_path = (
#         f"hub://{activeloop_username}/{args.activeloop_dataset_name}"
#     )

#     sys.argv = [
#         "streamlit",
#         "run",
#         "src/utils/chat.py",
#         "--",
#         f"--activeloop_dataset_path={args.activeloop_dataset_path}",
#     ]

#     sys.exit(stcli.main())


def main():
    """Define and parse CLI arguments, then execute the appropriate subcommand."""
    parser = argparse.ArgumentParser(description="Chat with a git repository")

    parser.add_argument(
        "--azure", action="store_true", help="Using Azure OpenAI API"
    )
    # parser.add_argument(
    #     "--include-file-extensions",
    #     nargs="+",
    #     default=None,
    #     help=(
    #         "Exclude all files not matching these extensions. Example:"
    #         " --include-file-extensions .py .js .ts .html .css .md .txt"
    #     ),
    # )

    parser.add_argument(
        "--repo-path",
        default="repos",
        help="The destination to clone the repository. Defaults to 'repos'.",
    )

    args = parser.parse_args()

    if args.azure:
        llm = AzureChatOpenAI(
            openai_api_base=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version="2023-05-15"
        )
    else:
        llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_ENDPOINT"), temperature=0.7)

    db = DB(args.repo_path)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.retriever)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot().style(height="750")
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            bot_message = generate_response(qa, history[-1][0])
            history[-1][1] = bot_message
            """"
            TODO: Streaming Feataures
            """
            # history[-1][1] = ""
            # for character in bot_message:
            #     history[-1][1] += character
            #     time.sleep(0.05)
            #     yield history

            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()