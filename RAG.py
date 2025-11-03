import os

from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

DEVICE = "mps:0"

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def get_embeddings(model_path):
    return HuggingFaceEmbeddings(model_name=model_path,
                                 model_kwargs={"device": DEVICE, "trust_remote_code": True})


def extract_docs(docs_path, glob, chunk_size):
    """"
    Extract all information in a given document directory and return in a Document object
    """
    if "pdf" in glob:
        loader = PyPDFDirectoryLoader(docs_path)
    else:
        loader = DirectoryLoader(docs_path, glob=glob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size / 10))
    documents = loader.load_and_split(text_splitter)
    return documents


def get_vector_store(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)


def generate_prompt_mixtral(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return (f"<s> [INST] {system_prompt} {prompt} [/INST]"
            f"Model Answer</s>").strip()


def generate_prompt_llama(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()


def get_llm(model_name, max_new_tokens=4096, temperature=0.5, top_p=0.95, repetition_penalty=1.15):
    return HuggingFaceEndpoint(
            endpoint_url=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            huggingfacehub_api_token=HF_TOKEN
            )


def get_conversation_chain(llm, vector_store):
    SYSTEM_PROMPT = (
        "Use the following pieces of context, along with the chat history if there is any, to answer the question at "
        "the end. If"
        "you don't know the answer, just say that you don't know, don't try to make up an answer.")

    template_mixtral = generate_prompt_mixtral("""
            {context}
            {chat_history}

            Question: {question}
            """.strip(), system_prompt=SYSTEM_PROMPT)

    prompt = PromptTemplate(template=template_mixtral, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            )


def get_conversation_chain_llama(llm, vector_store):
    SYSTEM_PROMPT = (
        "Use the following pieces of context, along with the chat history if there is any, to answer the question at "
        "the end. If"
        "you don't know the answer, just say that you don't know, don't try to make up an answer.")

    template_llama = generate_prompt_llama("""
            {context}
            {chat_history}

            Question: {question}
            """.strip(), system_prompt=SYSTEM_PROMPT)

    prompt = PromptTemplate(template=template_llama, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            )


def send_prompt_and_receive_answer(conversation_chain, prompt):
    response = conversation_chain.invoke(prompt)
    return response['answer']

# Unit1 is a vehicle with a 360 camera, 360 degree 3D lidar, and 4 60GHz receiver phased arrays and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter.
# Given the following information captured from unit1, describe the physical and network environment this information provides in detail,
# do your best to determine what objects and obstacles there are such as buildings and pedestrians,
# and do your best to estimate distances of objects and their types if there are any and how obstacles if any are affecting the network.
# The transmitter (Unit 2) is 8.066753628120216 meters away at a bearing of 182.27478933305133 degrees from the receiver (Unit 1).
# Objects detected using a LiDAR detection algorithm is as follows: A Vehicle is detected 15.07 meters away at a direction of 322.89 degrees. A Vehicle is detected 11.38 meters away at a direction of 229.71 degrees. A Vehicle is detected 9.83 meters away at a direction of 241.23 degrees. A Vehicle is detected 16.96 meters away at a direction of 181.47 degrees.
# Features extracted using PointNet from a 360 degree 3D LiDAR point cloud: -0.011237727478146553 -0.01121174544095993 0.05301650986075401 -0.09597904980182648 0.0906224399805069 0.027672799304127693 0.04369594156742096 0.001285133883357048 -0.01066519320011139 -0.07939758896827698 -0.0743304044008255 -0.0070366524159908295 -0.09438242018222809 0.05492638051509857 -0.011030086316168308 -0.03733726590871811 0.012949297204613686 -0.08458264917135239 -0.07961101830005646 -0.04842613264918327 -0.0020984942093491554 0.10580553114414215 -0.02934100851416588 -0.028767917305231094 -0.05254414677619934 -0.05755928158760071 -0.025427497923374176 0.010380707681179047 0.048186272382736206 -0.019428232684731483 -0.00913102738559246 0.08585315197706223 0.05513501167297363 0.10095472633838654 0.027947373688220978 0.004171017557382584 0.07287980616092682 0.0900825560092926 0.06811971962451935 -0.029096122831106186 0.003354049287736416 -0.02038336731493473 0.09012199938297272 0.04548356309533119 -0.017869018018245697 -0.06838048249483109 -0.007292289286851883 0.048617374151945114 0.037655945867300034 0.05620725080370903 -0.0555538535118103 -0.01760612428188324 0.0332752987742424 -0.0628386065363884 0.06316560506820679 -0.10024169087409973 0.0400555357336998 0.06960903108119965 0.08837508410215378 -0.03965676948428154 0.029138052836060524 -0.06754259020090103 -0.00844667386263609 -0.02483493834733963.
# Average measured power from front receiver is 0.00019585112363529333, the right receiver is 0.0002479094616774091, the back receiver is 0.03824070384052902, and the left receiver is 0.0001478471883729071.
