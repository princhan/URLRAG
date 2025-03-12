from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain import hub
from language_model import initialize_language_model

# Define a custom type for the agent's state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Invoke the agent model
def agent(state, tools):
    print("Call Agent: ANZ bank accounts")
    messages = state["messages"]
    model = initialize_language_model()
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

# Check the relevance of documents
def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("Check Relevance: Documents are relevant")

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = initialize_language_model().with_structured_output(Grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"]
    )
    chain = prompt | model
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    return "generate" if score == "yes" else "rewrite"

# Generate an answer based on the current state
def generate(state):
    print("Generate:")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    #prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate(
        template="""You are a bank analyst, you have to write all the accounts offered by the bank and its details properly as mentioned in the document \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n """,
        input_variables=["context", "question"]
    )

    llm = initialize_language_model()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# Transform the query to produce a better question
def rewrite(state):
    print("Rewrite: Transforming the query to produce a better question")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
        )
    ]
    model = initialize_language_model()
    response = model.invoke(msg)
    return {"messages": [response]}

# Create the state graph for the workflow
def create_graph(tools):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", lambda state: agent(state, tools))
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    return workflow.compile()
