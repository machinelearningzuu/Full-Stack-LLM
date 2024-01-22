import yaml, logging, sys, os
from llama_index.llms import AzureOpenAI, OpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import set_global_service_context, ServiceContext

logging.basicConfig(
                    stream=sys.stdout, 
                    level=logging.INFO
                    )
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

with open('cadentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

llm_flag = 'DIRECT'

embedding_llm = HuggingFaceEmbedding(
                                    model_name="BAAI/bge-small-en-v1.5",
                                    device='mps'
                                    )

if llm_flag == 'AZURE':
    llm=AzureOpenAI(
                    model=credentials['AZURE_ENGINE'],
                    api_key=credentials['AZURE_OPENAI_API_KEY'],
                    deployment_name=credentials['AZURE_DEPLOYMENT_ID'],
                    api_version=credentials['AZURE_OPENAI_API_VERSION'],
                    azure_endpoint=credentials['AZURE_OPENAI_API_BASE'],
                    temperature=0.3
                    )
    chat_llm = LLMPredictor(llm)
else:
    chat_llm = OpenAI(
                    api_key=credentials['DERMERZELAI_API_KEY'],
                    temperature=0.3
                    )
    
if llm_flag == 'AZURE':
    service_context = ServiceContext.from_defaults(
                                                    embed_model=embedding_llm,
                                                    llm_predictor=chat_llm
                                                    )
else:
    service_context = ServiceContext.from_defaults(
                                                    embed_model=embedding_llm,
                                                    llm=chat_llm
                                                    )

set_global_service_context(service_context)