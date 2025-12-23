from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from retrieval.retrievers import BM25Retriever, SparseRetriever
import requests
import json
import os
import time




def jlog(path, event, **k):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "event": event, **k}, ensure_ascii=False) + "\n")

TRACE = "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/searcher_trace.jsonl"





# add a query understanding step to the agent
SEARCH_SYSTEM_PROMPT = """You are an agent who's task is to find information that can help the user answer the given question. The user provides you with a question and some information and suggestions about what aspect to search for. This is a multi-step action. In this action, Step 1 is providing a search query that will help find relevant information. The user responds with search results. Then, in the Step 2 analyze the accuracy and relevance of the information. The user responds with confirmation about accuracy and relevance of the information. This continues until the provided information is sufficient.

# Step 1: 
## The user provides you with the following information as input
    - "question": the question the user wants to answer.
    - "information": a summary of the information the user has gathered so far. This can be empty if the user has not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.

### Output Schema:{{"search_query":<SEARCH_QUERY>,"search_query_explanation":<SEARCH_QUERY_EXPLANATION>}}
    - <SEARCH_QUERY> is the search query you suggest to find information that can help in answering the user's question.
    - <SEARCH_QUERY_EXPLANATION> is an explanation of why you suggest this search query.

    
    
# Step 2:
## your input: The user provides you with the search results.
"search_results": the search results using the query that you suggested.

### Output Schema: {{"query_id":<QUERY_ID>,"relevance":[{{"doc_id":<DOC_ID>,"is_relevant":<IS_RELEVANT>,"is_relevant_explanation":<IS_RELEVANT_EXPLANATION>}}],"change_search_query":<CHANGE_SEARCH_QUERY>,"new_search_query":<NEW_SEARCH_QUERY>,"end_search":<END_SEARCH>,"end_search_explanation":<END_SEARCH_EXPLANATION>}}
    - <QUERY_ID> is the query ID of the search result.
    - <DOC_ID> is The document ID of the search result that you want to verify.
    - <IS_RELEVANT> is a boolean value indicating whether the document is relevant to the query.
    - <IS_RELEVANT_EXPLANATION> is an explanation of why the document is relevant or not relevant.
    - <CHANGE_SEARCH_QUERY> is a boolean value indicating whether a new search query is needed to find relevant information. Setting it to True requires the "new_search_query" key's value to be updated and will initiate a new query with this value.
    - <NEW_SEARCH_QUERY> is the new search query used to find relevant information that can help answer the question. Is set to empty unless <CHANGE_SEARCH_QUERY> is set to True.
    - <END_SEARCH> is a boolean that represents satisfaction with the search results and that the search is complete. This will be False if the search needs to continue.
    - <END_SEARCH_EXPLANATION> is an explanation of why the search is complete.



"""

'''

For each of the input documents output exactly with this JSON Schema {{"query_id":<QUERY_ID>,"relevance":[{{"doc_id":"{ID}",  "is_relevant":<IS_RELEVANT(Document 1)>,"is_relevant_explanation":"<IS_RELEVANT_EXPLANATION(Document 1)>"}},{{"doc_id":"{ID_2}","is_relevant":<IS_RELEVANT(Document 2)>,"is_relevant_explanation":"<IS_RELEVANT_EXPLANATION(Document 2)>"}}],"change_search_query":<CHANGE_SEARCH_QUERY>,"new_search_query":<NEW_SEARCH_QUERY>,"end_search":<END_SEARCH>,"end_search_explanation":<END_SEARCH_EXPLANATION>}}


where:
    - <QUERY_ID> is the query ID of the search result.
    - <DOC_ID> is The document ID of the search result that you want to verify.
    - <IS_RELEVANT(Document 1)> is a boolean value indicating whether the Document 1 is relevant to the query.
    - <IS_RELEVANT_EXPLANATION(Document 1)> is an explanation of why Document 1 is relevant or not relevant.
    - <IS_RELEVANT(Document 2)> is a boolean value indicating whether the Document 2 is relevant to the query.
    - <IS_RELEVANT_EXPLANATION(Document 2)> is an explanation of why Document 2 is relevant or not relevant.
    - <IS_RELEVANT(Document 3)> is a boolean value indicating whether the Document 3 is relevant to the query.
    - <IS_RELEVANT_EXPLANATION(Document 3)> is an explanation of why Document 3 is relevant or not relevant.
    - <IS_RELEVANT(Document 4)> is a boolean value indicating whether the Document 4 is relevant to the query.
    - <IS_RELEVANT_EXPLANATION(Document 4)> is an explanation of why Document 4 is relevant or not relevant.
    - <IS_RELEVANT(Document 5)> is a boolean value indicating whether the Document 5 is relevant to the query.
    - <IS_RELEVANT_EXPLANATION(Document 5)> is an explanation of why Document 5 is relevant or not relevant.
'''



SEARCH_USER_PROMPT_TURN_1 = """# question: {QUESTION}"
# information: {INFORMATION}
# suggestions: {SUGGESTIONS}

Follow this Schema for the output, incorporating only the keys in this schema an no others:
Schema:{{"search_query":<SEARCH_QUERY>,"search_query_explanation":<SEARCH_QUERY_EXPLANATION>}}
    - <SEARCH_QUERY> is the search query you suggest to find information that can help in answering the user's question.
    - <SEARCH_QUERY_EXPLANATION> is an explanation of why you suggest this search query.
"""

SEARCH_USER_PROMPT_TURN_2 = """"This is the query from step 1 and the following documents returned from the query including their ids and texts:

Query ID: {QID}

Document 1 ID: {ID}
Document 1 text: {ANSWER}

Document 2 ID: {ID_2}
Document 2 text: {ANSWER_2}


Document 3 ID: {ID_3}
Document 3 text: {ANSWER_3}


Document 4 ID: {ID_4}
Document 4 text: {ANSWER_4}


Document 5 ID: {ID_5}
Document 5 text: {ANSWER_5}


For each of the input documents output exactly with this JSON Schema 
{{"query_id":<QUERY_ID>,"relevance"[{{"doc_id":"{ID}","is_relevant":<IS_RELEVANT(Document_i)>,"is_relevant_explanation":"<IS_RELEVANT_EXPLANATION(Document_i)>"}},...],"change_search_query":<CHANGE_SEARCH_QUERY>,"new_search_query":<NEW_SEARCH_QUERY>,"end_search":<END_SEARCH>,"end_search_explanation":<END_SEARCH_EXPLANATION>}}


where:
    - <QUERY_ID> is the query ID of the search result.
    - <DOC_ID> is The document ID of the search result that you want to verify.
    - <IS_RELEVANT(Document_i)> is a boolean value indicating whether the Document i is relevant to the query. i is document number 1 through 5.
    - <IS_RELEVANT_EXPLANATION(Document_i)> is an explanation of why Document i is relevant or not relevant. i is document number 1 through 5.
    - <CHANGE_SEARCH_QUERY> is a boolean value indicating whether a new search query is needed to find relevant information. Setting it to True requires the "new_search_query" key's value to be updated and will initiate a new query with this value.
    - <NEW_SEARCH_QUERY> is a new search query your formulate if <CHANGE_SEARCH_QUERY> above is True.
    - <END_SEARCH> is a boolean that marks the end of the search.
    - <END_SEARCH_EXPLANATION> is an explanation of why the search is ending.
    




"""

'''
SEARCH_USER_PROMPT_TURN_2 = (
    "This is the information resulting from your search query.\n"
    "Query ID: {QID}\n\n"
    "Document 1 ID: {ID_1}\n"
    "Document 1 text: {ANSWER_1}\n\n"
    "Document 2 ID: {ID_2}\n"
    "Document 2 text: {ANSWER_2}\n\n"
    "Reply with exactly one minified JSON object, then <ENDJSON>.\n"
    "Schema (exact): "
    '{{"query_id":"{QID}",'
    '"relevance":[{{"doc_id":"{ID_1}","is_relevant":true,"is_relevant_explanation":"<string>"}},'
                 '{{"doc_id":"{ID_2}","is_relevant":true,"is_relevant_explanation":"<string>"}}],'
    '"change_search_query":false,'
    '"new_search_query":"",'
    '"end_search":false,'
    '"end_search_explanation":"<string>"}}<ENDJSON>\n'
'''

def initilize_conversation():
    conversation = [
        {
            "role": "system",
            "content": SEARCH_SYSTEM_PROMPT
        }
    ]
    return conversation


def search(question, context, suggestions, memory, llm, retriever, execute_config):
    run_id     = execute_config.get("run_id")
    coord_turn = execute_config.get("coord_turn")

    # Ensure the container exists
    if 'searcher' not in memory:
        memory['searcher'] = {}

    s = memory['searcher']

    #fresh state per run (recommended), reset when run_id changes
    if s.get("run_id") != run_id:
        s.clear()
        s["run_id"]             = run_id
        s["conversation"]       = initilize_conversation()  # created once, only here
        s["queries"]            = {}
        s["id_to_query"]        = {}
        s["query_id"]           = 0
        s["verified_documents"] = []
        s["verified_ids"]       = []
        s["total_steps"]        = 0
    else:
        # Otherwise, *only* fill missing keys without re-calling heavy initializers
        if "conversation" not in s:
            s["conversation"] = initilize_conversation()
        if "queries" not in s:
            s["queries"] = {}
        if "id_to_query" not in s:
            s["id_to_query"] = {}
        if "query_id" not in s:
            s["query_id"] = 0
        if "verified_documents" not in s:
            s["verified_documents"] = []
        if "verified_ids" not in s:
            s["verified_ids"] = []
        if "total_steps" not in s:
            s["total_steps"] = 0

    # 2) global budget / deadline AFTER state is guaranteed
    if s["total_steps"] >= execute_config.get('searcher_steps_total_cap', 6):
        jlog(TRACE, "budget_stop", run_id=run_id, coord_turn=coord_turn,
             total_steps=s["total_steps"])
        return {
            "documents": [doc['text'] for doc in s['verified_documents']],
            "found_information": len(s['verified_documents']) > 0,
        }

    deadline_ts = execute_config.get('searcher_deadline_ts')
    if deadline_ts and time.time() >= deadline_ts:
        jlog(TRACE, "deadline_stop", run_id=run_id, coord_turn=coord_turn)
        return {
            "documents": [doc['text'] for doc in s['verified_documents']],
            "found_information": len(s['verified_documents']) > 0,
        }

    # Spend one global step for THIS call
    s["total_steps"] += 1




#end global budget
    '''
    if 'searcher' not in memory:
        memory['searcher'] = {
            "conversation": initilize_conversation(),
            "queries": {},
            "id_to_query": {},
            "query_id": 0,
            "verified_documents": [],
            "verified_ids": []
        }
    '''
    conversation = s["conversation"]
    
    #conversation = memory['searcher']['conversation']
    conversation.append({
        "role": "user",
        "content": SEARCH_USER_PROMPT_TURN_1.format(QUESTION=question, INFORMATION=context, SUGGESTIONS=suggestions)
    })
    if execute_config['agent_model_server']:
        conversation_text = conversation
    else:
        conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    #sampling_parmas = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_environment'], logprobs=1)
    sampling_parmas = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=500,stop=["<ENDJSON>"], logprobs=1)
    response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
    response_obj = str_to_json(response_text)
    
    jlog(TRACE, "searcher_step_1_prompt", run_id=run_id,coord_turn=coord_turn, raw_preview=conversation[-1]["content"][:1600])
    jlog(TRACE, "searcher_step_1_reply", run_id=run_id,coord_turn=coord_turn, raw_preview=response_text[:1600])
    
    
    
    #print(response_obj)
    conversation.append({
        "role": "assistant",
        "content": response_text
    })
    counter = 0
    query = response_obj['search_query']
    
    
    
    #verified_documents = memory['searcher']['verified_documents']
    #verified_ids = memory['searcher']['verified_ids']
    verified_documents = s["verified_documents"]
    verified_ids = s["verified_ids"]
    while execute_config['max_verifcation_same_query'] > counter:
        counter += 1
        jlog(TRACE, "searcher_loop_count",loop=counter,limit=execute_config['max_verifcation_same_query'])
        if query in memory['searcher']['queries']:
            query_id = memory['searcher']['queries'][query]['id']
        else:
            query_id = str(memory['searcher']['query_id'])
            memory['searcher']['queries'][query] = {
                "id": query_id,
                "query": query,
                "documents": {}
            }
            memory['searcher']['id_to_query'][query_id] = query
            memory['searcher']['query_id'] += 1
        if type(retriever) == SparseRetriever:
            document_1 = retriever.search_next(query)
            document_2 = retriever.search_next(query)
            document_3 = retriever.search_next(query)
            document_4 = retriever.search_next(query)
            document_5 = retriever.search_next(query)
        elif type(retriever) == BM25Retriever:
            document_1 = retriever.search_next(query)
            document_2 = retriever.search_next(query)
            document_3 = retriever.search_next(query)
            document_4 = retriever.search_next(query)
            document_5 = retriever.search_next(query)
            
            
            
            
            
            
        #increment initially never reached due to break
        '''  
        #document_1 = retriever.search_next(query)
        if document_1 is None:
            # no results for this query — consider reformulating the query or ending search
            break

        #document_2 = retriever.search_next(query)
        if document_2 is None:
            break
        '''
        if (document_1 is None) or (document_2 is None) or (document_3 is None) or (document_4 is None) or (document_5 is None):
            jlog(TRACE, "break",note='requests didn\'t return all of the documents')
            #counter+=1
            break
            
            

        #print("document from retriever",document_1)
        memory['searcher']['queries'][query]['documents'][document_1['id']] = {"doc": document_1, "verified": False}
        memory['searcher']['queries'][query]['documents'][document_2['id']] = {"doc": document_2, "verified": False}
        memory['searcher']['queries'][query]['documents'][document_3['id']] = {"doc": document_3, "verified": False}
        memory['searcher']['queries'][query]['documents'][document_4['id']] = {"doc": document_4, "verified": False}
        memory['searcher']['queries'][query]['documents'][document_5['id']] = {"doc": document_5, "verified": False}
        conversation.append({
            "role": "user",
            "content": SEARCH_USER_PROMPT_TURN_2.format(QID=query_id, ID=document_1['id'], ANSWER=document_1['text'].replace('"', "'"), ID_2=document_2['id'], ANSWER_2=document_2['text'].replace('"', "'"),ID_3=document_3['id'],ANSWER_3=document_3['text'].replace('"', "'"),ID_4=document_4['id'],ANSWER_4=document_4['text'].replace('"', "'"),ID_5=document_5['id'],ANSWER_5=document_5['text'].replace('"', "'"))
        })
        if execute_config['agent_model_server']:
            conversation_text = conversation
        else:
            conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
        
        
        jlog(TRACE, "searcher_step_2",raw_preview=response_text[:1600])
            
            
        #print(response_text)
        response_obj = str_to_json(response_text)
        #print(response_obj)
        conversation.append({
            "role": "assistant",
            "content": response_text
        })
        judgements = response_obj['relevance']
        for judgement in judgements:
            # print(memory['searcher']['queries'][query]['documents'].keys())
            doc_id = str(judgement['doc_id'])
            is_relevant = judgement['is_relevant']
            if is_relevant:
                memory['searcher']['queries'][query]['documents'][doc_id]['verified'] = True
                if doc_id not in verified_ids:
                    verified_ids.append(doc_id)
                    verified_documents.append(memory['searcher']['queries'][query]['documents'][doc_id]['doc'])
        
        change_search_query = response_obj['change_search_query']  
        if change_search_query:
            query = response_obj['new_search_query']
            jlog(TRACE, "break",note='changing query')
            #changing the search query is taking another step so increment
            #counter+=1
            continue
        is_ending_search = response_obj['end_search']
        if is_ending_search:
            jlog(TRACE, "break",note='ending search')
            #counter +=1
            break
        #counter += 1
    return {
        "documents": [doc['text'] for doc in verified_documents],
        "found_information": len(verified_documents) > 0
    }
