from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Components
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Input
topic = "Machine Learning"

chain = RunnableParallel({
    "short" :RunnableLambda(lambda x :x['short']) |short_prompt | model | parser ,
    "detailed" :RunnableLambda(lambda x: x['detailed']) |detailed_prompt |model |parser
})

#runable parallel takes a dict of runnables and runs them in parallel, it takes a dict as input and passes the relevant part of the dict to each runnable
#runable lambda is used to extract the relevant part of the input dict for each runnable

result = chain.invoke({
    "short" : {"topic":"Machine Learning"},
    "detailed" : {"topic":"Deep Learning"}
})

print(result['short'])
print(result['detailed'])