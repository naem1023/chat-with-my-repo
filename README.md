# Chat with my repository

**!! Still in development. Insufficient functionality !!**  
Develop while talking to your code repository.

## Features
### LLM Support
- [x] Azure OpenAI API, OpenAI API
- [ ] Huggingface model
### Conversation
- [x] Langchain RetrievalConversatinoChain
- [ ] Langchain Conversation Agent for Chat Models

### DB
- [x] FAISS
- [ ] Milvus-lite


## Requirements
- Install dependency
- Add your ".env"
```sh
# python=3.11

pip install --upgrade poetry
poetry install
```

## Run

```sh
python main.py --repo-path <your_repo_path> --azure
```