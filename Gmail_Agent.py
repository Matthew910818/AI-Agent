import os
import uuid
import json
from langchain_community.agent_toolkits import GmailToolkit 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.tools import BaseTool
from typing import Optional, Type, List, Dict, Any
from openai import OpenAI
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

# -------------------------------
# INITIALIZATION & SETUP
# -------------------------------
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = QdrantClient(":memory:")

# Create a collection for email memory if it doesn't exist
collection_name = "email_memory"
collections = client.get_collections().collections
collection_names = [collection.name for collection in collections]
if collection_name not in collection_names:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# Initialize OpenAI embeddings and vector store
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -------------------------------
# SET UP GMAIL TOOLKIT
# -------------------------------
toolkit = GmailToolkit()

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)
gmail_tools = toolkit.get_tools()

# -------------------------------
# SET UP THE LLM & OTHER TOOLS
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------
# CUSTOM TOOLS FOR MEMORY & WEB SEARCH
# -------------------------------
class SaveToMemoryTool(BaseTool):
    name: str = "save_to_memory"
    description: str = "Save important information to the agent's memory for future reference"
    
    def _run(self, information: str) -> str:
        doc_id = str(uuid.uuid4())
        vectorstore.add_texts(
            texts=[information],
            metadatas=[{"source": "user_input", "id": doc_id}],
            ids=[doc_id]
        )
        return f"Information saved to memory with ID: {doc_id}"
    
    def _arun(self, information: str) -> str:
        raise NotImplementedError("This tool does not support async")

class RetrieveFromMemoryTool(BaseTool):
    name: str = "retrieve_from_memory"
    description: str = "Retrieve information from the agent's memory using a query"
    
    def _run(self, query: str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        if results:
            return "\n\n".join([doc.page_content for doc in results])
        else:
            return "No relevant information found in memory."
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

class WebSearchTool(BaseTool):
    name: str = "web_search_preview"
    description: str = "Search the internet for up-to-date information not available in the agent's knowledge base."
    
    def _run(self, query: str) -> str:
        try:
            completion = openai_client.responses.create(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": query}],
                tools=[{"type": "web_search_preview"}]
            )
            search_result = completion.output_text
            
            doc_id = str(uuid.uuid4())
            vectorstore.add_texts(
                texts=[f"Web search for '{query}': {search_result}"],
                metadatas=[{"source": "web_search", "query": query, "id": doc_id}],
                ids=[doc_id]
            )
            
            return search_result
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# -------------------------------
# NEW TOOL: PROCESS INSURANCE CLAIM
# -------------------------------
class ProcessInsuranceClaimTool(BaseTool):
    name: str = "process_insurance_claim"
    description: str = (
        "Process an insurance claim email by reading, gathering details from memory and the web, "
        "and drafting a well-written response for the claim. If the email contains ambiguous or incomplete details, "
        "automatically invoke the web search tool to gather additional guidelines."
    )

    def _run(self, claim_query: str) -> str:
        # Step 1: Use the Gmail search tool to look for relevant insurance claim emails.
        search_tool = next((tool for tool in gmail_tools if tool.name == "search_gmail"), None)
        if search_tool is None:
            return "Error: Gmail search tool not found."
        # For example, search by keyword "insurance" plus the claim_query.
        search_response = search_tool.run(f"insurance {claim_query}")
        
        # Assume search_response is a list of email objects and pick the first one.
        if isinstance(search_response, list) and len(search_response) > 0:
            email_id = search_response[0].get('id', None)
            if not email_id:
                return "No valid email ID found in search response."
        else:
            return "No insurance email found."
        
        # Step 2: Read the email content using the Gmail read tool.
        read_tool = next((tool for tool in gmail_tools if tool.name == "get_gmail_message"), None)
        if read_tool is None:
            return "Error: Gmail read tool not found."
        read_response = read_tool.run(email_id)
        # Assume the read_response is a dict containing the email body.
        email_content = read_response.get('body', '')
        
        # Step 3: Save the key details from the email to memory.
        SaveToMemoryTool()._run(email_content)
        
        # Step 4: Retrieve similar past cases from memory.
        past_cases = RetrieveFromMemoryTool()._run(claim_query)
        
        # Step 5: Assess email ambiguity.
        # If the email contains ambiguous details (e.g. the word "experimental" without explanation),
        # then invoke the web search tool to gather additional context.
        additional_info = ""
        if "experimental" in email_content.lower():
            additional_info = WebSearchTool()._run(claim_query)
        
        # Step 6: Combine the gathered information.
        combined_info = (
            f"Email content:\n{email_content}\n\n"
            f"Past successful cases:\n{past_cases}\n\n"
            f"Additional info from web search:\n{additional_info}"
        )
        
        # Step 7: Use the LLM to draft a response email based on the combined information.
        draft_response = llm(
            [{"role": "user", "content": f"Draft a detailed, professional response email for an insurance claim based on the following details:\n\n{combined_info}"}]
        ).content
        
        # Step 8: Send the drafted response using the Gmail send tool.
        send_tool = next((tool for tool in gmail_tools if tool.name == "send_gmail_message"), None)
        if send_tool is None:
            return "Error: Gmail send tool not found."
        send_tool.run(f"Send the following email: {draft_response}")
        
        return f"Insurance claim processed. Email sent with the following response:\n{draft_response}"

    def _arun(self, claim_query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# -------------------------------
# COMBINE ALL TOOLS & SET UP THE AGENT
# -------------------------------
memory_tools = [SaveToMemoryTool(), RetrieveFromMemoryTool()]
web_tools = [WebSearchTool()]
process_tools = [ProcessInsuranceClaimTool()]

all_tools = gmail_tools + memory_tools + web_tools + process_tools

# Update your instructions to reflect the new capabilities and self-assessment
instructions = """
You are a medical insurance assistant that helps doctors process insurance claims.
Your tasks include:
1. Reading incoming emails from Gmail related to insurance claims.
2. Saving important claim details to memory.
3. Retrieving past successful cases from memory.
4. Automatically assessing the completeness of the email details. If you encounter ambiguous or incomplete information (for example, if an email simply states that a treatment is "experimental" without further explanation), you must use the web_search_preview tool to gather additional up-to-date guidelines and similar cases.
5. Drafting a well-written, professional response email for the insurance claim.
6. Sending the response back to the insurance company.

When processing an insurance claim, use the following tools as needed:
- 'gmail_search' to find relevant emails.
- 'gmail_read' to read email content.
- 'save_to_memory' to store key details.
- 'retrieve_from_memory' to recall similar cases.
- 'web_search_preview' to gather additional information if you detect ambiguity or missing details.
- 'process_insurance_claim' to execute the full claim response workflow.
"""

base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    memory=memory,
)

# -------------------------------
# INTERACTION FUNCTION
# -------------------------------
def interact_with_agent(query: str) -> str:
    response = agent_executor.invoke({"input": query})
    return response["output"]

# -------------------------------
# MAIN LOOP (for testing)
# -------------------------------
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Agent: Goodbye!")
            break
        response = interact_with_agent(user_input)
        print(f"Agent: {response}")