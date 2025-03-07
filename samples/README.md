# Running Langchain OPEA SDK with OPEA microservices

The OPEA Langchain SDK facilitates effortless interaction with open-source large language models, such as Llama 3, directly on your local machine. To leverage the SDK, you need to deploy an OpenAI compatible model serving.
This local microservice deployment is crucial for harnessing the power of advanced language models while ensuring data privacy, reducing latency, and providing the flexibility to select models without relying on external APIs.

## 1. Starting the microservices using compose

A prerequisite for using Langchain OPEA SDK is that users must have OpenAI compatible LLM text/embeddings generation service (etc., TGI, vLLM) already running. Langchain OPEA SDK package uses these deployed endpoints to help create end to end enterprise generative AI solutions.

This approach offers several benefits:

Data Privacy: By running models locally, you ensure that sensitive data does not leave your infrastructure.

Reduced Latency: Local inference eliminates the need for network calls to external services, resulting in faster response times.

Flexibility: You can bring your own OPEA validated [models](https://github.com/opea-project/GenAIComps/blob/main/comps/llms/src/text-generation/README.md#validated-llm-models) and switch between different models as needed, tailoring the solution to your specific requirements.

To run the services, set up the environment variables:

```bash
export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
export HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN
export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
```

Run Docker Compose:

```bash
docker compose up -d
```

## 2. Check the services are up and running

```bash
curl ${host_ip}:6006/v1/embeddings \
    -X POST \
    -d '{"input":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

```bash
# TGI service
curl http://${host_ip}:9009/v1/chat/completions \
    -X POST \
    -d '{"model": "Intel/neural-chat-7b-v3-3", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17}' \
    -H 'Content-Type: application/json'
```

## 3. Install Langchain OPEA package

You can install LangChain OPEA package in several ways:

### Install from Source

To install the package from the source, run:

```bash
pip install poetry && poetry install --with test
```

### Install from Wheel Package

To install the package from a pre-built wheel, run:

1. **Build the Wheels**: Ensure the wheels are built using Poetry.
   ```bash
   poetry build
   ```
2. **Install via Wheel File**: Install the package using the generated wheel file.
   ```bash
   pip install dist/langchain_opea-0.1.0-py3-none-any.whl
   ```

## 4. Install Jupyter Notebook

```bash
pip install notebook
```

## 5. Run the samples notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the [`summarize.ipynb`](./summarize.ipynb) notebook and run the cells to execute the summarization example.

Open the [`qa_streaming.ipynb`](./qa_streaming.ipynb) notebook and run the cells to execute the QA chatbot with retrieval example.
