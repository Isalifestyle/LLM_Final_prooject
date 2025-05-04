from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
# from chatreadretrieveread import *
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
import ast
import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv

#start from here

file_path = r"sample_HPS_data_HFM_PHD.csv"

df_pd = pd.read_csv(file_path)

# Clean and parse the embedding column
df_pd = df_pd[["chunk_id", "file_name", "embedding", "file_content"]].dropna()

df_pd["embedding"] = df_pd["embedding"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Convert to matrix
embedding_matrix = np.array(df_pd["embedding"].tolist()).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Save index to disk
faiss.write_index(index, "my_faiss.index")

# Save metadata (e.g. chunk_id, file_name, file_content)
df_pd[["chunk_id", "file_name", "file_content"]].to_parquet("metadata.parquet", index=False)


load_dotenv(override=True)
os.environ['DATABRICKS_TOKEN'] = os.getenv('DATABRICKS_TOKEN')
os.environ['DATABRICKS_HOST'] = os.getenv('DATABRICKS_HOST')

chat_model = ChatDatabricks(
    endpoint="gpt-4o-mini",
    temperature=0.1,
    max_tokens=2000,
)

find_exact_keyword = ChatDatabricks(
    endpoint="gpt-4o-mini",
    temperature=0.1,
    max_tokens=2000,
)
embedding_model = DatabricksEmbeddings(
    endpoint = "ada-002"
)

#chat_model.invoke("Do you know of a method, which does not involve a total rebuild, for changing the ‘ExpSQLSvc’ and ‘ExpSQLAgtSvc’ accounts from local to domain users, please?")
def get_embeddings(embedding_model, query):
    embedding = embedding_model.embed_query(query)
    return embedding

def retrieve_chunks(index, query, k=5):
    query_vector = get_embeddings(embedding_model, query)
    query_vector = np.array(query_vector).astype("float32").reshape(1,-1)

    distance, indices = index.search(query_vector, k)

    retrieved_chunks = df_pd.iloc[indices[0]]

    return retrieved_chunks


def get_keywords_for_exact_search(query):
       
        system_messsage_keyword_extraction = """
 
        # Advanced Keyword Extraction for Technical Queries
 
        You are an advanced AI assistant specialized in extracting crucial keywords from technical queries. Your expertise lies in pinpointing precise product information, including names, unique identifiers, and version details.
 
        ## Key Elements to Extract:
 
        - **Product Name:** The primary identifier or common name of a product (e.g., Experion PKS).
        - **SKU/Part Number:** Unique alphanumeric codes used to distinguish products or parts (e.g., RDYNAMO-8074, 1-EVDYZ2U).
        - **Version/Release Info:** Specific version numbers or release identifiers for products or documents (e.g., R2.2.1, Update4).
 
        ## Extraction Guidelines:
 
        - **Priority:** Focus on extracting the product name first, followed by any specific identifiers (SKU/Part Number) or version/release information.
        - **Patterns:** Pay attention to alphanumeric strings and version formats, which may include numbers, letters, hyphens, and periods.
        - **Ignore Non-essential Words:** Overlook filler words such as "is," "in," "for," etc., unless they are part of a product name or identifier.
        - **Special Cases:** Treat terms like "C300" or "301C" as product names when not accompanied by other identifiable product names.
 
        ## Task Execution:
 
        1. Carefully review each query to discern the primary focus.
        2. Extract relevant keywords, focusing on product names, part numbers/SKUs, and version/release information.
        3. Present the extracted keywords separated by commas. For queries that focus on a specific functionality or general inquiry without mentioning a product/version, use "no keyword found"
 
        ## Enhanced Examples:
 
        - **Input Query:** "Is there a Tools and Controller Update4 for R520.2?"
        - **Expected Keywords:** "Tools and Controller, Update4, R520.2"
        - **Input Query:** "What product anomalies are being introduced in Honeywell Forge Alarm Management Reporting R2.2.1?"
        - **Expected Keywords:** "Honeywell Forge Alarm Management Reporting, R2.2.1"
        - **Input Query:** "C300 sync steps for CBM6 v2"
        - **Expected Keywords:** "C300, CBM6 v2"
        - **Input Query:** "what is scada status in QuickBuilder?"
        - **Expected output:** "no keyword found"
        - **Input Query:** "C300 sync steps"
        - **Expected Keywords:** "C300"
 
       
        **Input:**
 
        {raw_query}
        """
 
        messages= [
            {"role": "system", "content": system_messsage_keyword_extraction.format(raw_query=query)},
            ]
 
        print('generating keywords: ', chat_model)
        completion = find_exact_keyword.invoke(messages).content
 
        return completion

#retrieve_chunks(index, "nUniformance PHD 200")['file_content'].to_list()

#build the core question
hcot_query = """
    You are an expert in understanding and rephrasing user service queries based on chat history. Your goal is to break down the user's intent and context into a concise, reformulated question. 

    Internal Thought Process:

    1. Analyze the user's current query {input}: Carefully examine the user's latest input to understand their core need and any specific details they have provided.
    2. Consider the chat history: Review the provided {chat_history} to identify any relevant prior interactions. Play close attention to the flow of the conversation and the assitant's previous responses. If there is no chat history or if the chat history is not related to the user's current query, consider the query as a standalone query. 
    3. Identify the user's underlying intent: Based on current query and relevant history, determine what the user is really trying to achieve or understand. 
    4. Generate a context-rich question: Formulate a new, standalone question that accurately reflect's the user's intent, incorporatin grelevant context from the chat history. Ensure that this generated question is not a direct follow-up or answer to the current user query. 
    5. Refine the question: Ensure that the new question is clear, concise, relevant and in a proper question format. Avoid any repetition or ambiguity. 

    Output: Only the reformulated user question should be provided below

    Based on the chat history: {chat_history}

    What is the user's core question, taking into account the user's previous conversation?
    """

core_question_prompt = PromptTemplate(
        template = hcot_query,
        input_variables=["chat_history", "input"]
)

query = "Do you know of a method, which does not involve a total rebuild, for changing the ‘ExpSQLSvc’ and ‘ExpSQLAgtSvc’ accounts from local to domain users, please?"
core_question_chain = core_question_prompt | chat_model
core_question = core_question_chain.invoke({"chat_history": "", "input":query}).content
core_question
#use the core question built to retrieve documents or just their file content?
#build the user query
retr_docs = retrieve_chunks(index, core_question)['file_content'].to_list()
user_query = core_question +  " ".join(retr_docs)
prompt_prefix = """Respond to the user query marked by #### based on the context encapsulated within ```.
    - Your response should strictly utilize the information from the provided Sources in line with user query.
    - Every piece of information or fact you use from a source must be immediately cited within the sentence itself, using its "fileName", DO NOT USE ANY OTHER NAME. Example: "The sky is blue [info.txt]."
    - DO NOT STATE CITATIONS at the very end of all the facts.
    - If a single fact is backed by multiple sources, integrate each citation within the same sentence: "The sky is blue [info1.txt][info2.txt]."
    - After formulating your answer, double-check for any inconsistencies or omitted citations.

    STRICTLY frame your response in JSON using the following fields, do not :
    - "response": The comprehensive answer to the user query with inline citations after each fact. For multiple lines, give a "\n" seperated answer.
    - "confidence": Quantify your confidence in the provided response on a numeric scale.
    - "reason": Briefly (in no more than 20 words) elucidate the rationale behind your answer.

    User query: ####{user_query}####
    
    Response: """ 

final_prompt = PromptTemplate(
    template = prompt_prefix,
    input_variables = ["user_query"]
) 

final_response_chain = final_prompt | chat_model
final_reponse = final_response_chain.invoke({"user_query": user_query})
final_reponse
def extract_response_from_string(json_string):
    json_content = json_string.strip('```').strip()
    json_pattern = re.compile(r'json\s*({.*?})', re.DOTALL)
    match = json_pattern.search(json_content)
    if match:
        json_obj = json.loads(match.group(1))
        return json_obj["response"]
    return None
    

extract_response_from_string(final_reponse.content)