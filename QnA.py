from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings import CohereEmbeddings
from langchain.chat_models import ChatOpenAI

import cohere
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import RetrievalQA
co_embeddings =CohereEmbeddings(cohere_api_key='api-key',model= "multilingual-22-12")
co = cohere.Client("your-apikey")

import pinecone



def qa(q,num=1):
    print(num)
    pinecone.init(api_key="api-key", environment="us-west4-gcp-free")
    index_table_name = 'chat-qa-wikipedia'

    index = pinecone.Index(index_table_name)

    text_field = "text"


    docs =  Pinecone(
        index, co_embeddings.embed_query, text_field
    )

    querys = str(q)

    doc = docs.similarity_search(
        querys, 
        k=int(num)
    )
   
    return doc

def chatqa(q):
    llm = ChatOpenAI(
    openai_api_key='api-key',
    model_name='gpt-3.5-turbo',
    temperature=0.0
        ) 
    doc =qa(q)
    print(doc)
    chain = load_qa_chain(llm, chain_type="stuff")
    query = q
    result2 = chain.run(input_documents=doc, question=query)  

    return result2
