import cohere
import openai
import pinecone
from fastapi import FastAPI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pydantic import BaseModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
import re

app = FastAPI()

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "f7668e4e-8594-4e11-8850-b272faf7d492"
PINECONE_ENV = "us-west4-gcp"
co = cohere.Client("GP6aGX4AlXzTE6wru1cSWqXbcCegq3Ey8L45Ls78")

openai.api_key = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "evidgpt"
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

uri = "mongodb+srv://Datainput:inputdata@cluster0.jpw0qjv.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi("1"))

db = client["cmedai"]  # replace with your database name
collection = db["教科书"]  # replace with your collection name


class Query(BaseModel):
    query: str


@app.post("/syndrome")
def syndrome(query: Query):
    # Perform NER
    ner_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """為症狀進行NER, 请只按例子output NER的内容，不需要症状NER以外任何东西

            例子：
            input
            - 患者最近情况好一点了，已经转阴。
            - 患者在日本时自己发烧，回香港后看中医。
            - 中医给患者开的中药可以舒缓症状，有抗病毒功效。
            - 患者有些痰，喉咙有点干，胃口正常，睡眠质量好。
            - 患者咳嗽两天，有些感冒，鼻塞、鼻水倒流，喉嚨有些痛和乾，有痰黏著喉嚨，大便正常，睡眠六至七个小时，舌头有点红，吃过炸鸡。
            - 中医师会给患者开药治疗感冒
output：
            痰，喉嚨干，感冒，鼻塞，咳嗽""",
            },
            {
                "role": "user",
                "content": query.query,
            },  # Use query.query instead of ner_output
        ],
    )

    # Assign the ner_output after the completion
    ner_output = ner_completion.choices[0].message["content"]

    # Perform similarity search
    retrieved_docs = vectorstore.similarity_search(
        query=ner_output, k=30, namespace="syndromes"
    )
    retriever_texts = [doc.page_content for doc in retrieved_docs]

    # Perform cohere rerank
    response = co.rerank(
        model="rerank-multilingual-v2.0",
        query=ner_output,
        documents=retriever_texts,
        top_n=5,
    )

    # Combine all retrieved texts into a single string
    retriever_texts = [f"{doc.document['text']}" for doc in response]
    combined_text = " ".join(retriever_texts)

    # Parse the combined text
    reranked_texts = re.sub(r"(症狀: .*?)\s*(證型: .*?證)\s*", r"\2\n\1\n\n", combined_text)

    # Return the parsed text after reranking
    return reranked_texts



@app.post("/lifestyle")
def lifestyle(query: Query):
    pipeline = [
        {
            "$search": {
                "index": "diseaseName",
                "autocomplete": {"query": query.query, "path": "病名"},
            }
        },
        {"$limit": 1},
        {"$project": {"_id": 0, "病名": 1, "预防与调摄": 1}},
    ]
    result = collection.aggregate(pipeline)
    return list(result)


@app.post("/prescription")
def prescription(query: Query):
    # Perform NER
    ner_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """為症狀進行NER, 请只按例子output NER的内容，不需要症状NER以外任何东西

            例子：
            input
            - 患者最近情况好一点了，已经转阴。
            - 患者在日本时自己发烧，回香港后看中医。
            - 中医给患者开的中药可以舒缓症状，有抗病毒功效。
            - 患者有些痰，喉咙有点干，胃口正常，睡眠质量好。
            - 患者咳嗽两天，有些感冒，鼻塞、鼻水倒流，喉嚨有些痛和乾，有痰黏著喉嚨，大便正常，睡眠六至七个小时，舌头有点红，吃过炸鸡。
            - 中医师会给患者开药治疗感冒
output：
            痰，喉嚨干，感冒，鼻塞，咳嗽""",
            },
            {
                "role": "user",
                "content": query.query,
            },
        ],
    )

    # Assign the ner_output after the completion
    ner_output = ner_completion.choices[0].message["content"]

    # Perform similarity search
    retrieved_docs = vectorstore.similarity_search(
        query=ner_output, k=50, namespace="fangji"
    )
    retriever_texts = [doc.page_content for doc in retrieved_docs]

    # Perform cohere rerank
    response = co.rerank(
        model="rerank-multilingual-v2.0",
        query=ner_output,
        documents=retriever_texts,
        top_n=5,
    )

    # Return the entire content of the response
    return response.__dict__


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
