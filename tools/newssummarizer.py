from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool=TavilySearchResults(max_results=5)

model=ChatMistralAI(model="mistral-small-latest", temperature=0.9)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes news articles."),
    ("human", "Summarize the following news article: {article}")
])

parser=StrOutputParser()

chain =prompt|model | parser




result=search_tool.run("What are the latest news articles about artificial intelligence?")

prompt.format_messages(article=result)
output = chain.invoke({"article": result})

print(output)



