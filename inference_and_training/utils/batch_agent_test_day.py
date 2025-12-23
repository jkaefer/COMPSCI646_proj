from agents.coordinator.agent import generate_response
from configs.default import DEFAULT_CONFIG_2, DEFAULT_CONFIG
import json
import pandas as pd
import argparse
import concurrent.futures
from utils.general import batchify
import tqdm
import json
import copy
import time
import os
from transformers import AutoTokenizer

def remove_log(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

#updated, difference is not returning response with ground_truth(unseen so far)
def run_agent(query, config):
    """
    Run the agent with the given query and configuration.
    """
    counter = 0
    errors = []
    config = copy.deepcopy(config)
    saved_result = None
    while counter < config['max_retries']:
        try:
            response = generate_response(query, config)
            response['memory']['generator']
            saved_result = response
            assert len(response['verified_documents']) > 0, "No documents found in the response."
            #return {
            #    "question": query,
            #    "response": response,
            #    "ground_truth": answer,
            #    "success": True
            #}
            return {
                "question": query,
                "response": response,
                "success": True
            }
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            errors.append(str(e))
            counter += 1
            if config['temperature_agent'] < 1:
                config['temperature_agent'] += 0.1
            if config['temperature_environment'] < 1:
                config['temperature_environment'] += 0.1
    if saved_result:
        #return {
        #    "question": query,
        #    "response": saved_result,
        #    "ground_truth": answer,
        #    "success": True,
        #    "error": errors
        #}
        return {
            "question": query,
            "response": saved_result,
            "success": True,
            "error": errors
        }
    return {
        "question": query,
        "response": None,
        "success": False,
        "error": errors
    }

parser = argparse.ArgumentParser(description="Run the batch agent")
parser.add_argument("--queries_addr", type=str, required=True)
parser.add_argument("--our_id_to_fineweb_id_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--max_workers", type=int, default=8, help="Number of threads to use for parallel processing")
parser.add_argument("--num_config", default=1, type=int, help="Configuration to use for the agent (1 or 2)")
parser.add_argument("--agent_name", type=str, default="", help="Name of the agent to use")
parser.add_argument("--no_concise", action="store_true", help="Indicate if the agent should be concise")





# -----------------------
# Input loading (JSON array or JSONL fallback) chatgpt ##########################
# -----------------------

def load_queries_any(path: str) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Returns:
      queries: List[str]
      ids:     List[str]
      answers_refs: List[List[str]]  (each: list of reference answers)
      gold_docs:    List[List[str]]
    Supports:
      - JSON array (your format)
      - JSONL with {"id","question"} lines (legacy)
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            queries, ids, answers_refs, gold_docs = [], [], [], []
            for item in data:
                queries.append(item.get("question", ""))
                ids.append(str(item.get("id", "")))
                ans_list = item.get("answer", [])
                
                
                #why is this the case here? why determine if string or list?
                if isinstance(ans_list, list):
                    refs = [str(x) for x in ans_list]
                elif isinstance(ans_list, str):
                    refs = [ans_list]
                else:
                    refs = []
                answers_refs.append(refs)
                gdocs = item.get("gold_documents", [])
                if not isinstance(gdocs, list):
                    gdocs = []
                gold_docs.append([str(x) for x in gdocs])
            return queries, ids, answers_refs, gold_docs
            
        #the alternative is an empty list, empty list from test.json
        else:
            queries, ids, answers_refs, gold_docs = [], [], [], []
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    queries.append(obj.get("question", ""))
                    ids.append(str(obj.get("id","")))
                    answers_refs.append([])
                    gold_docs.append([])
            return queries, ids, answers_refs, gold_docs
            
########END of input loading helper function ###########################


####CHATGPT METRIC HELPER FUNCTIONS ##########################

#splitting up prediction into 
def _tokenize(s: str) -> List[str]:
    return s.lower().strip().split()

#checking if predicted is in references
def exact_match(pred: str, references: List[str]) -> float:
    p = (pred or "").strip().lower()
    refs = [(r or "").strip().lower() for r in (references or [])]
    return 1.0 if p in refs else 0.0


#
def token_f1(pred: str, references: List[str]) -> float:
    # standard SQuAD-style token F1, max over references
    p_toks = _tokenize(pred)
    #there are no relative references
    if not references:
        return 0.0
    best = 0.0
    from collections import Counter
    pc = Counter(p_toks)
    for ref in references:
        rc = Counter(_tokenize(ref))
        
        #overlap between prediction tokens and reference tokens
        # actually performing multiset (bag) intersection, which is:
        # pc & rc  = returns a Counter whose counts are the minimum of each token’s counts in pc and rc
        #the sum is the total number of overlapping tokens between the two strings
        overlap = sum((pc & rc).values())
        if overlap == 0:
            continue
        prec = overlap / (sum(pc.values()) + 1e-8)
        rec  = overlap / (sum(rc.values()) + 1e-8)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
        
    #the best f1 score for the overlap between a reference of the references and the prediction
    return best





def is_faithful_by_docs(verified_doc_ids: List[str], gold_doc_ids: List[str], k: int = 1) -> float:
    # simple binary: at least k verified docs appear in gold set
    s_ver = set(filter(None, verified_doc_ids or []))
    s_gold = set(filter(None, gold_doc_ids or []))
    return 1.0 if len(s_ver & s_gold) >= k else 0.0
    
    
    
#######END OF METRIC HELPER FUNCTIONS #########################################################################################################





if __name__ == "__main__":

    log_paths = ["/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/coordinator_trace.jsonl",
        "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/answerer_trace.jsonl",
        "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/searcher_trace.jsonl",
        "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/server_llm_trace.jsonl"]
    for log in log_paths:
        remove_log(log)
        
        
    args = parser.parse_args()
    
    
    queries, ids, answers_refs, gold_docs = load_queries_any(args.queries_addr)
    
    #queries_addr = args.queries_addr

    # Load the queries from the CSV file
    
    # json file in the established pipeline 646 final project

    
    results = []
    start_time = time.time()
    
    #how a batch size is decided
    total_batches = len(queries) // args.max_workers + (1 if len(queries) % args.max_workers > 0 else 0)
    
    
    #update to the following
    #queries, ids, answers_refs, gold_docs
    for batch_queries, batch_ids, batch_answers_refs, batch_gold_docs in tqdm.tqdm(zip(batchify(queries, args.max_workers), batchify(ids, args.max_workers), batchify(answers_refs, args.max_workers),batchify(gold_docs, args.max_workers), total=total_batches):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            counter = 0
            for query, answer in zip(batch_data, batch_ground_truth):
                if args.num_config == 1:
                    config = DEFAULT_CONFIG
                elif args.num_config == 2:
                    if counter % 2 == 0:
                        config = copy.deepcopy(DEFAULT_CONFIG)
                    else:
                        config = copy.deepcopy(DEFAULT_CONFIG_2)
                else:
                    raise ValueError("Invalid configuration number. Use 1 or 2.")
                if args.agent_name:
                    config["agent_model"] = args.agent_name
                if args.no_concise:
                    config["concise"] = False
                #run_agent no longer passed the answer
                futures.append(executor.submit(run_agent, query, config))
                counter += 1
                
            #MAJOR CHANGE: not converting to liverag format    
            #for id, result in zip(batch_ids, futures):
            #    output = result.result()
            #    output["id"] = id
            #    results.append(output)
            for cur_id, query, answer_refs, gold_docs, future in zip(batch_ids, batch_queries, batch_refs, batch_gold_docs, futures):
                output = future.result()
                item = { 
                    "id": cur_id,
                    "question": query,
                    "success": bool(output.get("success")),
                    "error": output.get("error", []),
                    "response_text": "",
                    "verified_documents": [],
                    "gold_documents": gold_docs,
                    "metrics": {}
                }
                    
                #MAJOR CHANGE: Computing metrics right now, vs by judge of competition
                #check to see if response is not null/None and success is true
                if output.get('success') and output.get('response'):
                    response = output.get('response')
                    #determine verified documents, these were determined by the 'Searcher' AgentType, see coordinator/agent.py file
                    #verified_documents and their doc_ids are determined on lines 143 and 144 of searcher/agent.py
                    #the searcher agent has previously determined through relevance judgements 1 per document if documents are relevant
                  
                    #KEY:doc_id very original source is from the corpus used to 'train' the retriever
                    #doc_id e.g. is "A_BOTTLE_OF_Old_Wine_1"
                    item['verified_documents'] = [document.get("doc_id", "") for document in response.get("verified_documents", [])]
                    
                    
                    #Metrics computation section
                    #passing verified document ids, as well as goldend documents
                    prf = grounding_prf(item['verified_documents'],item['gold_documents'])
                    
                    #answer_refs stand for answer_references
                    exact_match = exact_match(item['response_text'], answer_refs)
                    f1 = token_f1(item['response_text'],answer_refs)
                    
                    #for now faithfulness is very lenient, if only one verified_docs appears in the gold set then passes
                    faithful = is_faithful_by_docs(item['verified_documents'], item['gold_documents'],1)
                    
                    #commiting metrics to item
                    item["metrics"] = { 
                        "grounding": prf,
                        "correctness":{'em':exact_match,'token_f1':f1},
                        "faithfulness":{"is_faithful":faithful}
                    }
                #append to results list
                results.append(item)
    
    #write result metrics to results file
    with open(args.output_addr, "w") as out_f:
        for item in results:
            out_f.write(json.dumps(item) + "\n")
                   
                    
        #save as csv, but not liverag format         
        #convert_to_live_rag_format_and_save(results, args.output_addr, our_id_to_fineweb_id)
        
        
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
