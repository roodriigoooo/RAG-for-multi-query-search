from langchain.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import bs4

os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

# define OPENAI API KEY here
# define LANGCHAIN API KEY here


os.environ['LANGCHAIN_PROJECT']="HELPFUL: RAG-for-Multi-Query-Search"


#this is just for demonstration purposes, in our case we would use the JSONLoader

#loader = JSONLoader(file_path='path_to_your_json_files', jq_schema = '.[]')
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/", ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=('post-content', 'post_title', 'post-header')
        )
    ),
)
blog_docs = loader.load()

#split into chunks, if applicable
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap = 40)
#splitting
splits = text_splitter.split_documents(blog_docs)

# set to true : os.environ['LANGCHAIN_TRACING_V2']
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

# define OPENAI API KEY here
# define LANGCHAIN API KEY here

os.environ['LANGCHAIN_PROJECT']="RAG-for-Multi-Query-Search"


vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# Multi Query: Different Perspectives
template = """Your task is to generate five different versions of the given user demanded ingredients, to retrieve the most
relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search, and help them find a document that includes all 
the ingredients, or as many of them as possible. Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What grocery store that is near me can provide me with walnuts, food storage bags, and kosher salt"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
# Tesing a single retriever
# docs = retrieval_chain.invoke({"question":question})
# len(docs)

template = """Retrieve the most relevant documents from the following request based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})

## here is the output generated

# Although I am unsure of its location, GenericGrocery can provide you with your requested ingredients. Unfortunately,
# no other grocery store has these in stock, altough RobDessertKitchen is expected to have these in inventory.