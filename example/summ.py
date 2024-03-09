import os
from dotenv import load_dotenv

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from esg_toolkit.log_utils import logger
from esg_toolkit.data_utils import read_data, save_data
from esg_toolkit.qdrant_utils import QdrantDBWrapper
from esg_toolkit.curation_utils import to_embeddings

import sys
sys.path.append("..")
from prompt import map_reduce_prompt


load_dotenv()
show_query_process = True
streaming = False

text_splitter = RecursiveCharacterTextSplitter()

llm = AzureChatOpenAI(
    temperature=0, model_name=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
    deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    streaming=streaming, callbacks=[StreamingStdOutCallbackHandler()],
)

qdrant_executor = QdrantDBWrapper()

res = qdrant_executor.scroll(
    collection_name="greenhousegas", limit=10, scroll_filter=None
)[0]
res = [r.payload for r in res if "查詢過於頻繁" not in r.payload["content"]]

text = res[0]["content"]

texts = text_splitter.split_text(text)

docs = [Document(page_content=t) for t in texts]

chain = load_summarize_chain(
    llm, chain_type="map_reduce", 
    map_prompt=map_reduce_prompt.PROMPT,
    combine_prompt=map_reduce_prompt.PROMPT,
    verbose=show_query_process
)

output = chain.run(docs)

print(output, flush=True)

