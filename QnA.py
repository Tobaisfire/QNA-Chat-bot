from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings import CohereEmbeddings
from langchain.chat_models import ChatOpenAI

import cohere
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import RetrievalQA
co_embeddings =CohereEmbeddings(cohere_api_key='PoQqB6c283yGmex4A2cSwQWxYj5oP1rh9bkuqKYy',model= "multilingual-22-12")
co = cohere.Client("PoQqB6c283yGmex4A2cSwQWxYj5oP1rh9bkuqKYy")

import pinecone



def qa(q,num=1):
    print(num)
    pinecone.init(api_key="f6e73bf8-43dc-4ce4-b29b-19430caa8543", environment="us-west4-gcp-free")
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
    openai_api_key='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',
    model_name='gpt-3.5-turbo',
    temperature=0.0
        ) 
    doc =qa(q)
    print(doc)
    chain = load_qa_chain(llm, chain_type="stuff")
    query = q
    result2 = chain.run(input_documents=doc, question=query)  

    return result2
