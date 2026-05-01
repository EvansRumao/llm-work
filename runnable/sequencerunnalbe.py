from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

input = "What is the capital of France?"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke(input)

print(response)
