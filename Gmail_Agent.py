import os
import uuid
import json
import datetime
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

# Create collection for medical terminology
medical_collection_name = "medical_terminology"
if medical_collection_name not in collection_names:
    client.create_collection(
        collection_name=medical_collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# Initialize OpenAI embeddings and vector store
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

medical_vectorstore = Qdrant(
    client=client,
    collection_name=medical_collection_name,
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
    client_secrets_file="client_secret_159062750550-nhseqhjcn29g9o6lc9vric0g5ali9uk0.apps.googleusercontent.com.json",
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
            metadatas=[{"source": "user_input", "id": doc_id, "timestamp": datetime.datetime.now().isoformat()}],
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
    name: str = "web_search"
    description: str = """Search the internet for up-to-date information. Use this tool when:
1. You need to find current facts, news, or information not in your knowledge base
2. You need current date, time, weather, or other time-sensitive information
3. You need to clarify ambiguous terms, acronyms, or technical jargon in emails
4. You need to verify claims or statements made in emails
5. You need context about organizations, people, or products mentioned in emails
6. You need to research industry trends or background information relevant to email content
7. You need to research insurance policies, medical procedures, or billing codes"""
    
    def _run(self, query: str) -> str:
        try:
            # Use OpenAI's gpt-4o-search-preview model for web search
            completion = openai_client.chat.completions.create(
                model="gpt-4o-search-preview",
                web_search_options={
                    "search_context_size": "medium",  # Options: "low", "medium", "high"
                },
                messages=[{
                    "role": "user",
                    "content": query,
                }],
            )
            
            # Extract the search results
            search_result = completion.choices[0].message.content
            
            # Save search result to memory for future reference
            doc_id = str(uuid.uuid4())
            vectorstore.add_texts(
                texts=[f"Web search for '{query}': {search_result}"],
                metadatas=[{"source": "web_search", "query": query, "id": doc_id, "timestamp": datetime.datetime.now().isoformat()}],
                ids=[doc_id]
            )
            
            return search_result
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

class TimeInfoTool(BaseTool):
    name: str = "get_current_time_info"
    description: str = """Get current date, time, and related information. Use this when you need accurate current time information such as:
1. Current date
2. Current time
3. Day of week
4. Month or year
5. Any other time-sensitive information that requires up-to-date knowledge"""
    
    def _run(self, query: str) -> str:
        """Use web search to get current time information."""
        search_query = f"current {query} today"
        try:
            # Delegate to WebSearchTool to get current information
            web_tool = WebSearchTool()
            result = web_tool._run(search_query)
            return result
        except Exception as e:
            return f"Error retrieving time information: {str(e)}"
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

class MedicalTerminologyTool(BaseTool):
    name: str = "medical_terminology_lookup"
    description: str = """Look up or verify medical terminology, procedure codes, diagnosis codes, or billing information. 
Use this tool when:
1. You need to verify a medical term or its definition
2. You need to check ICD-10 or CPT codes for procedures/diagnoses
3. You need to understand medical abbreviations
4. You need accurate, standard medical language for insurance claims
5. You need to verify standard of care for specific conditions"""
    
    def _run(self, query: str) -> str:
        try:
            # First check if we have this in our specialized medical memory
            results = medical_vectorstore.similarity_search(query, k=1)
            if results and len(results) > 0:
                return f"From medical database: {results[0].page_content}"
                
            # If not found in memory, search the web
            search_query = f"medical terminology {query} healthcare definition"
            web_tool = WebSearchTool()
            result = web_tool._run(search_query)
            
            # Save this to our medical terminology collection
            doc_id = str(uuid.uuid4())
            medical_vectorstore.add_texts(
                texts=[f"Medical term: {query}\nDefinition: {result}"],
                metadatas=[{"term": query, "id": doc_id, "timestamp": datetime.datetime.now().isoformat()}],
                ids=[doc_id]
            )
            
            return result
        except Exception as e:
            return f"Error looking up medical terminology: {str(e)}"
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

class DraftProfessionalMessageTool(BaseTool):
    name: str = "draft_professional_medical_claim"
    description: str = """Draft a professional, compliant medical insurance claim message based on provided information.
This tool specializes in crafting clear, persuasive, and technically accurate messages for insurance claims. 
Use when you need to create formal communication to insurance companies."""
    
    def _run(self, input_data: str) -> str:
        """
        Input should be a JSON string with the following fields:
        {
            "claim_details": "Description of the claim",
            "patient_info": "Relevant non-PHI patient information",
            "procedure_info": "Description of procedures/treatments",
            "insurance_policy": "Relevant insurance policy details",
            "prior_communications": "Summary of previous communications if any"
        }
        """
        try:
            # Parse input data
            data = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            # Construct a prompt for drafting a professional message
            prompt = f"""
            Draft a professional medical insurance claim message using the following information:
            
            CLAIM DETAILS: {data.get('claim_details', 'Not provided')}
            
            PATIENT INFORMATION: {data.get('patient_info', 'Not provided')}
            
            PROCEDURE INFORMATION: {data.get('procedure_info', 'Not provided')}
            
            INSURANCE POLICY DETAILS: {data.get('insurance_policy', 'Not provided')}
            
            PRIOR COMMUNICATIONS: {data.get('prior_communications', 'Not provided')}
            
            The message should:
            1. Be highly professional and formal
            2. Include all relevant medical terminology accurately
            3. Clearly reference policy provisions that support the claim
            4. Be concise but comprehensive
            5. Follow standard medical billing communication practices
            6. Include any relevant procedure/diagnosis codes if available
            7. Avoid emotional appeals and focus on facts and policy
            """
            
            # Generate the professional draft
            completion = llm.invoke(prompt)
            draft = completion.content
            
            # Save this draft to memory
            SaveToMemoryTool()._run(f"DRAFTED CLAIM:\n{draft}")
            
            return draft
            
        except Exception as e:
            return f"Error drafting professional message: {str(e)}"
    
    def _arun(self, input_data: str) -> str:
        raise NotImplementedError("This tool does not support async")

class QualityAssessmentTool(BaseTool):
    name: str = "assess_draft_quality"
    description: str = """Evaluate the quality and effectiveness of a drafted insurance claim message.
Use this tool to check if a draft is ready to be sent or needs improvement.
The assessment will check for professionalism, compliance with medical standards, clarity, and persuasiveness."""
    
    def _run(self, draft: str) -> str:
        try:
            prompt = f"""
            Evaluate this medical insurance claim message draft for quality and effectiveness:
            
            DRAFT MESSAGE:
            {draft}
            
            Please assess the following criteria and provide a detailed evaluation:
            
            1. PROFESSIONALISM (1-10): Is the language formal, appropriate for a professional medical context?
            2. TECHNICAL ACCURACY (1-10): Does it use medical terminology correctly and appropriately?
            3. POLICY RELEVANCE (1-10): Does it clearly reference relevant insurance policy provisions?
            4. CLARITY (1-10): Is the message clear and unambiguous?
            5. COMPLETENESS (1-10): Does it include all necessary information for the claim?
            6. PERSUASIVENESS (1-10): Does it make a compelling case for approval?
            7. COMPLIANCE (1-10): Does it comply with standard healthcare communication practices?
            
            For each category, provide a specific explanation of strengths and weaknesses.
            
            Then provide:
            
            OVERALL SCORE (1-10): 
            
            RECOMMENDED IMPROVEMENTS:
            
            FINAL ASSESSMENT: Is this draft ready to send (YES/NO)?
            """
            
            # Generate the assessment
            completion = llm.invoke(prompt)
            assessment = completion.content
            
            return assessment
            
        except Exception as e:
            return f"Error assessing draft quality: {str(e)}"
    
    def _arun(self, draft: str) -> str:
        raise NotImplementedError("This tool does not support async")

# -------------------------------
# ENHANCED INSURANCE CLAIM PROCESSING TOOL
# -------------------------------
class ProcessInsuranceClaimTool(BaseTool):
    name: str = "process_insurance_claim"
    description: str = (
        "Process a medical insurance claim by gathering relevant information, drafting a professional response, "
        "assessing its quality, and sending it after approval. This tool handles the end-to-end claim process "
        "and will automatically determine when additional research or verification is needed."
    )

    def _run(self, claim_query: str) -> str:
        # Step 1: Use the Gmail search tool to look for relevant insurance claim emails.
        search_tool = next((tool for tool in gmail_tools if tool.name == "search_gmail"), None)
        if search_tool is None:
            return "Error: Gmail search tool not found."
        # Search by keyword "insurance" plus the claim_query.
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
        
        # Step 5: Use the LLM to assess if the email needs additional research
        assessment_prompt = f"""
        Analyze this medical insurance claim email content and determine if additional research is needed.
        
        Email content:
        {email_content}
        
        Context from memory:
        {past_cases}
        
        Should I perform additional web research to better understand this claim? 
        Answer with only YES or NO, followed by a brief explanation (max 1 sentence).
        """
        
        assessment_response = llm.invoke(assessment_prompt).content.strip()
        
        # Extract the YES/NO decision from the response
        additional_info = ""
        medical_terminology_info = ""
        if assessment_response.upper().startswith("YES"):
            # The LLM decided that additional research is needed
            # Generate a relevant search query based on the email content
            search_query_prompt = f"""
            Based on this medical insurance claim email, generate a specific web search query to gather 
            additional information needed to process the claim properly.
            
            Email content:
            {email_content}
            
            Formulate a specific, targeted search query (max 15 words).
            """
            
            search_query = llm.invoke(search_query_prompt).content.strip()
            additional_info = WebSearchTool()._run(search_query)
            
            # Step 5b: Check if there are medical terms that need clarification
            medical_terms_prompt = f"""
            From the following email content, extract any medical terminology, procedure codes, diagnosis codes,
            or technical medical terms that might need verification for an insurance claim.
            
            Email content:
            {email_content}
            
            Additional information:
            {additional_info}
            
            List only the medical terms that need verification, one per line. If none, respond with "NONE".
            """
            
            medical_terms_response = llm.invoke(medical_terms_prompt).content.strip()
            
            if not medical_terms_response.upper() == "NONE":
                # We have medical terms to verify
                terms = medical_terms_response.split("\n")
                medical_terminology_info = ""
                for term in terms[:3]:  # Limit to first 3 terms to avoid excessive lookups
                    if term.strip():
                        term_info = MedicalTerminologyTool()._run(term.strip())
                        medical_terminology_info += f"\n\nTerm: {term}\nInformation: {term_info}"
            
            # Save the fact that we did this research
            SaveToMemoryTool()._run(f"Web research performed for claim: {search_query}\nResults: {additional_info}")
            if medical_terminology_info:
                SaveToMemoryTool()._run(f"Medical terminology verified: {medical_terminology_info}")
        
        # Step 6: Prepare data for message drafting
        claim_info = {
            "claim_details": claim_query,
            "patient_info": email_content,  # This would be parsed more carefully in a real implementation
            "procedure_info": medical_terminology_info if medical_terminology_info else "Not specified",
            "insurance_policy": "Reference policy information from email",
            "prior_communications": past_cases
        }
        
        # Step 7: Draft a professional message
        draft_json = json.dumps(claim_info)
        draft_response = DraftProfessionalMessageTool()._run(draft_json)
        
        # Step 8: Assess the quality of the draft
        quality_assessment = QualityAssessmentTool()._run(draft_response)
        
        # Step 9: Determine if the draft is ready to send
        if "FINAL ASSESSMENT: YES" in quality_assessment.upper():
            # The draft is ready to send
            send_ready = True
            final_draft = draft_response
        else:
            # The draft needs improvement
            improvement_prompt = f"""
            Based on this quality assessment, improve the draft message:
            
            ORIGINAL DRAFT:
            {draft_response}
            
            QUALITY ASSESSMENT:
            {quality_assessment}
            
            Please provide an improved version that addresses all the identified issues.
            """
            
            improved_draft = llm.invoke(improvement_prompt).content
            
            # Assess the improved draft
            improved_assessment = QualityAssessmentTool()._run(improved_draft)
            
            # Determine if the improved draft is ready
            if "FINAL ASSESSMENT: YES" in improved_assessment.upper():
                send_ready = True
                final_draft = improved_draft
            else:
                # If after improvement it's still not ready, we'll send it with a note
                send_ready = True
                final_draft = improved_draft + "\n\nNote: This draft may need further review."
        
        # Step 10: Send the final message if it's ready
        if send_ready:
            send_tool = next((tool for tool in gmail_tools if tool.name == "send_gmail_message"), None)
            if send_tool is None:
                return "Error: Gmail send tool not found."
            
            # In a real implementation, we would properly format the message with recipient, subject, etc.
            send_tool.run(f"Send the following email: {final_draft}")
            
            return f"""
            Insurance claim processed and sent. 
            
            FINAL MESSAGE:
            {final_draft}
            
            QUALITY ASSESSMENT:
            {quality_assessment}
            """
        else:
            return f"""
            Insurance claim processed but requires manual review before sending.
            
            DRAFT MESSAGE:
            {draft_response}
            
            QUALITY ASSESSMENT:
            {quality_assessment}
            """

    def _arun(self, claim_query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# -------------------------------
# COMBINE ALL TOOLS & SET UP THE AGENT
# -------------------------------
memory_tools = [SaveToMemoryTool(), RetrieveFromMemoryTool()]
web_tools = [WebSearchTool(), TimeInfoTool()]
medical_tools = [MedicalTerminologyTool(), DraftProfessionalMessageTool(), QualityAssessmentTool()]
process_tools = [ProcessInsuranceClaimTool()]

all_tools = gmail_tools + memory_tools + web_tools + medical_tools + process_tools

# Update your instructions to reflect the specialized medical insurance focus
instructions = """You are a specialized medical insurance assistant that helps doctors process insurance claims.
You have access to email, memory storage, web search, and specialized medical and insurance tools.

Your primary capabilities include:
1. Reading and analyzing medical insurance emails
2. Researching medical terminology and insurance policies
3. Storing important claim information in memory
4. Retrieving similar past cases from memory
5. Web searching for current medical guidelines and policies
6. Drafting professional medical insurance claim messages
7. Assessing the quality of draft messages for compliance and effectiveness
8. Sending polished, professional responses to insurance companies

When processing insurance claims:
1. Understand what the doctor or medical office is asking for
2. Use the most appropriate tools based on the specific claim needs
3. Make your own decisions about which tools to use - don't wait for specific instructions
4. Be proactive in researching medical terminology and policy details
5. Use your memory to build context from past claims
6. Always assess the quality of drafted messages before sending them

Guidelines for tool selection:
- Use Gmail tools for email operations (search, read, send)
- Use memory tools to store and retrieve information about past claims
- Use the medical_terminology_lookup tool for verifying medical terms, ICD/CPT codes
- Use the draft_professional_medical_claim tool to create compliant, effective messages
- Use the assess_draft_quality tool to evaluate all drafts before sending
- Use the get_current_time_info tool for any date-related questions
- Use web search for researching:
  * Current medical guidelines
  * Insurance policy details
  * Precedent cases for similar claims
  * Procedure and billing information

IMPORTANT: 
1. NEVER use your built-in knowledge for time-sensitive information like current dates or times.
2. ALWAYS assess the quality of drafted messages before sending them.
3. Medical accuracy and compliance are your highest priorities.

You have full autonomy to decide which tools to use based on the specific claim at hand.
"""

base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
agent = create_openai_functions_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,  # Set to True so we can see the agent's thought process
    memory=memory,
)

# Example interaction function
def interact_with_agent(query: str) -> str:
    response = agent_executor.invoke({"input": query})
    return response["output"]

# Add this line if you want to run the agent from command line
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Agent: Goodbye!")
            break
        response = interact_with_agent(user_input)
        print(f"Agent: {response}")