import cohere
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = FastAPI()

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "f7668e4e-8594-4e11-8850-b272faf7d492"
PINECONE_ENV = "us-west4-gcp"
co = cohere.Client("GP6aGX4AlXzTE6wru1cSWqXbcCegq3Ey8L45Ls78")
uri = "mongodb+srv://Datainput:inputdata@cluster0.jpw0qjv.mongodb.net/?retryWrites=true&w=majority"
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


client = MongoClient(uri, server_api=ServerApi('1'))
openai.api_key = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "evidgpt"
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


class Query(BaseModel):
    query: str

###


db = client['cmedai']  # replace with your database name
collection = db['教科书']  # replace with your collection name

def autocomplete_search(query, collection, path="病名", index="diseaseName"):
    pipeline = [
        {"$search": {
            "index": index, 
            "autocomplete": {
                "query": query, 
                "path": path
            }
        }},
        {"$limit": 1},
        {"$project": {"_id": 0, "病名": 1, "预防与调摄": 1}},  
    ]
    result = collection.aggregate(pipeline)
    return list(result)

@app.get("/autocomplete_search/{query}")
async def autocomplete_search_route(query: str):
    try:
        result = autocomplete_search(query, collection)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search")
def search(query: Query):
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
    print(f"NER Output: {ner_output}")

    ner_output = ner_completion.choices[0].message["content"]
    print(f"NER Output: {ner_output}")

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

    # Use the combined text as the prompt for the chat completion
    prompt = """從以下內容，一步一步地辨別，以及患者的症狀與哪一個證型類似，按照可能性由高到低排列

    回答方式：
    證型1： {{證型}}
    進一步鑒別:{{為該證型症状中，患者沒有的症状。 例子：“ 詢問有無：小便失禁，耳鳴等症狀”}}
    相似症狀：{{原因：一步一步解釋為什麼你判斷患者的症状與這個證型相似。例如： “心悸，氣短，自汗，這些都是__證常見的症狀”}}

    （剩餘證型必須按照以上格式以及方式表達，必須回答最少4個證型）
    證型2：
    進一步鑒別：
    相似症狀：

    證型3：
    進一步鑒別：
    相似症狀：

    證型4：
    進一步鑒別：
    相似症狀：

    參考內容：
    {}""".format(
        combined_text
    )

    print(combined_text)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ner_output},
        ],
        temperature=0,
    )

    print("Prompt tokens: ", completion["usage"]["prompt_tokens"])
    print("Completion tokens: ", completion["usage"]["completion_tokens"])
    print("Total tokens: ", completion["usage"]["total_tokens"])
    return completion.choices[0].message["content"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
