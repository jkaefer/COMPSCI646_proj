from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json
from retrieval.retrievers import BM25Retriever, SparseRetriever
from enum import Enum
from agents.generator.agent import generate_answer, revise_answer
from agents.summarizer.agent import generate_summary
from agents.reasoner.agent import generate_analysis
from agents.validator.agent import validate_response
from agents.searcher.agent import search
from agents.complex_searcher.agent import generate_subquestions
from agents.planner.agent import generate_plan
import json
from utils.server_llm import SeverLLM, load_url_from_log_file

import os
import time

def jlog(path, event, **k):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "event": event, **k}, ensure_ascii=False) + "\n")
        

TRACE = "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/coordinator_trace.jsonl"


class AgentType(Enum):
    VALIDATOR = "validator"
    SUMMARIZER = "summarizer"
    PLANNER = "planner"
    REASONER = "reasoner"
    COMPLEX_SEARCHER = "complex_searcher"
    ANSWERER = "answerer"
    REVISER = "reviser"
    SEARCHER = "searcher"
    FINISHER = "finisher"
    
    

COORDINATOR_SYSTEM_PROMPT_NEGATIVE_INFORMED = """You are a highly capable agent, and your goal is to  generate a concise response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is concise, informative, and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.

## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

#Take a look at this negative example and think about why this example strategy failed, the "verified_documents" included in the negative example are the attempt the agent made to guess the "gold_documents" also included in the negative example:
# negative example: {NEG_ROUTE}

"""

COORDINATOR_SYSTEM_PROMPT_POSITIVE_INFORMED = """You are a highly capable agent, and your goal is to  generate a concise response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is concise, informative, and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.

## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

#Take a look at this positive example and think about why this example strategy worked, the "verified_documents" included in the positive example are the attempt the agent made to guess the "gold_documents" also included in the positive example:
# positive example: {POS_ROUTE}


"""



COORDINATOR_SYSTEM_PROMPT_POSITIVE_INFORMED = """You are a highly capable agent, and your goal is to  generate a concise response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is concise, informative, and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.

## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

#Take a look at this positive example and think about why this example strategy worked, the "verified_documents" included in the positive example are the attempt the agent made to guess the "gold_documents" also included in the positive example:
# positive example: {POS_ROUTE}


"""


COORDINATOR_SYSTEM_PROMPT = """You are a highly capable agent, and your goal is to  generate a concise response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is concise, informative, and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.

## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

"""




COORDINATOR_SYSTEM_PROMPT_COMPLEX = """You are a highly capable agent, and your goal is to  generate a concise response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is concise, informative, and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.


## complex_searcher: This agent can help decide whether the main question should be split into multiple simpler sub-questions that. It then uses either the main question or sub-questions in a search for information to help answer the main question. You need to use this agent at least once in the task in order to find documents that help answer the question.
### input:
    - "question": the original question the user wants to answer.
    - "information": the information you have gathered so far. This can be empty.
### output:
    - "should_split": a boolean. If false, you think the original question should be treated as a single question.
    - "sub_questions": a list of sub-questions. Each item in the list should be a JSON object with:
        - "id": an integer identifier for the sub-question.
        - "sub_question": the text of the sub-question.
        - "reason": a short explanation of why this sub-question is useful for answering the original question.


## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.



## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

"""

COORDINATOR_SYSTEM_PROMPT_NO_CONCISE = """You are a highly capable agent, and your goal is to  generate a response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is informative and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.



## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.

## searcher: This agent can help you search for information that can help you answer the given question. You can use this agent to search for information that can help you answer the question. You can use this agent multiple times in any time during the task to search for information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "suggestions": a set of suggestions about what aspect to search for.
### output:
    - "found_information": a boolean value indicating whether the search found relevant information.
    - "documents": a list of documents that the agent has found that are relevant to the search query.

## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

"""






COORDINATOR_SYSTEM_PROMPT_NO_CONCISE_COMPLEX = """You are a highly capable agent, and your goal is to  generate a response to the given question. This is a multi-turn task and you don't have to do it in a single turn. You have access to a set of capable agents, each with a specific skill, and you can choose the most suitable agent for each turn to help you in generating the response. In each turn, you should select the most suitable agent from a provided list of agents to help you generate the response. The agents can help you in different aspects such as validating the response, searching for information, analyzing the information, and summarizing the information. Your goal is to generate a response that is informative and relevant to the question. You can use the agents to help you in different aspects of generating the response. You can also use the agents multiple times to generate the response. You can also use the agents in any order you like. To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. The list of agents and their input and output formats are provided below:

# Agents:

## validator: This agents can help you verify if the generated response meets the criteria for the given question. 
### input:
    - "question": the question the user wants to answer.
    - "information": the information the you has gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "response": the response generated to the question.
### output:
    - "extracted_criteria": a list of criteria that are extracted from only the user's question (e.g., being detailed, concise, short, long, etc.), containing the following fields:
        - "criteria": the criteria extracted from the user's question.
        - "criteria_explanation": an explanation of why extracted this criteria.
        - "is_response_valid": a boolean value indicating whether the response is valid according to the extracted criteria.
        - "is_response_valid_feedback": feedback on whether the response is valid according to the extracted criteria and how it can be improved.
    - "is_groundedly_supported": a boolean value indicating whether the all parts of the response is grounded with supporting information.
    - "is_groundedly_supported_feedback": feedback on whether the response is grounded with supporting information and how it can be improved.
    - "is_correctly_answered": a boolean value indicating whether the response is correct.
    - "is_correctly_answered_feedback": feedback on whether the response is correct and how it can be improved.

## summarizer: This agent can help you summarize the information you have gathered so far.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to summarize. This can be empty if you have not gathered any information yet.
### output:
    - "summary": the summary of the information that the agent has generated.

## planner: This agent can help you plan a strategy to generate the response to the given question. It is suggested to use this agent at the beginning of the task to plan the strategy. You can also use this agent multiple times in any time during the task to plan the strategy.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to share with the agent. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
### output:
    - "plan": the plan that the agent has generated to generate the response to the given question.

## reasoner: This agent can help you reason about the information you have gathered so far about specific aspects of the question. You can use this agent to reason about the information you have gathered if you need help with reasoning about the information.
### input:
    - "question": the question the user wants to answer.
    - "information": the information you have gathered so far and want to reason about. This can be a summary or highlights of the information you have gathered so far and does not need to be the full information. This can be empty if you have not gathered any information yet.
    - "aspect": the aspect of the question you want to reason about.
### output:
    - "analysis": the reasoning about the information that the agent has generated.


## complex_searcher: This agent can help decide whether the main question should be split into multiple simpler sub-questions that. It then uses either the main question or sub-questions in a search for information to help answer the main question.
### input:
    - "question": the original question the user wants to answer.
    - "information": the information you have gathered so far. This can be empty.
### output:
    - "should_split": a boolean. If false, you think the original question should be treated as a single question.
    - "sub_questions": a list of sub-questions. Each item in the list should be a JSON object with:
        - "id": an integer identifier for the sub-question.
        - "sub_question": the text of the sub-question.
        - "reason": a short explanation of why this sub-question is useful for answering the original question.


## answerer: This agent can help you generate the response to the given question. You can use this agent to generate the response to the question. You can use this agent multiple times in any time during the task to generate the response.
### input:
    - "question": the question the user wants to answer.
    - "guidance": a guidance on how should the agent structure its response and what to include in this response. This should help the agent to generate a better response based on the information you have gathered so far, but it should not be the answer itself.
    - "important_information": a string that outlines the most important information that should be included in the response.
### output:
    - "response": the response that the agent has generated to the given question.

## reviser: This agent can help you revise the response generated by the answerer agent. You can use this agent to revise the response generated by the answerer agent if you need help with revising the response. Note that you cannot use this agent before answerer agent.
### input:
    - "question": the question the user wants to answer.
    - "suggestion": a string that outlines the suggested revisions to the response.
### output:
    - "response": the revised response that the agent has generated.


## finisher: You can end the task by using this agent. By selecting this agent, you indicate that you have finished the task and the latest response generated by the answerer or reviser agent is the final response to the question.
### input:
    - "finished": a boolean value indicating that you have finished the task.
### output: the agent will not provide any output.
### Note: You should only provide this input to the agent in the given format and you don't need to provide the response to the agent. 

# question: {QUESTION}

"""





COORDINATOR_USER_PROMPT = """To choose an agent, you need to provide the a valid json object in ```json ``` block that contains the following fields:
    - "agent": the name of the agent you want to choose. 
    - "input": the input you want to provide to the agent. This should be a valid json object that contains the required fields for the chosen agent.
    - "reason": a brief explanation of why you chose the agent for the given input.
In response, the agent will provide you with the output in the specified format for the chosen agent. In selecting an agent, you should select the agent that you think is the most appropriate to take next. Using the same agent multiple times is allowed if you think it is necessary, but might not be useful always. In response, the agent will provide you with the necessary information to continue the conversation.
"""

def initilize_conversation(question, concise=True):
    conversation = [
        {
            "role": "system",
            "content": COORDINATOR_SYSTEM_PROMPT.format(QUESTION=question) if concise else COORDINATOR_SYSTEM_PROMPT_NO_CONCISE.format(QUESTION=question)
        }
    ]
    return conversation
    
def initilize_conversation_complex(question, concise=True):
    conversation = [
        {
            "role": "system",
            "content": COORDINATOR_SYSTEM_PROMPT_COMPLEX.format(QUESTION=question) if concise else COORDINATOR_SYSTEM_PROMPT_NO_CONCISE.format(QUESTION=question)
        }
    ]
    return conversation
    


def parse_experience_json(experience_path):

    json_dict = None
    with open(experience_path, 'r') as f:
        json_dict = json.load(f)

    pos_memory = None
    pos_gold_docs = None

    neg_memory = None
    neg_gold_docs = None

    for question_ind in json_dict.keys():
        faithfulness = json_dict[question_ind][0]['metrics']['metric']['faithful_score']['scor_faithfulness']


        if faithfulness == 1.0:
            pos_memory = json_dict[question_ind][0]['response']['agent_conversation_text']
            pos_verified_docs = json_dict[question_ind][0]['response']['verified_documents']
            pos_gold_docs = json_dict[question_ind][0]['gold_documents']
        else: 
            neg_memory = json_dict[question_ind][0]['response']['agent_conversation_text']
            neg_verified_docs = json_dict[question_ind][0]['response']['verified_documents']
            neg_gold_docs = json_dict[question_ind][0]['gold_documents']
		
        if (pos_memory != None) and (neg_memory != None):
            break
			
    return ([pos_memory,pos_verified_docs,pos_gold_docs],[neg_memory,neg_verified_docs,neg_gold_docs])
		
		
def initilize_conversation_informed(question, concise,experience_path):
    pos, neg = parse_experience_json(experience_path)
    
    pos = json.dumps(pos)
    neg = json.dumps(neg)
    conversation = [
        {
            "role": "system",
            "content": COORDINATOR_SYSTEM_PROMPT_NEGATIVE_INFORMED.format(QUESTION=question,NEG_ROUTE=neg)
        }
    ]
    return conversation

def initilize_memory():
    memory = {
        "responses" : [],
        "final_response": None
    }
    return memory

def update_conversation(conversation, role, content):
    conversation.append({
        "role": role,
        "content": content
    })
    return conversation

def get_next_user_prompt(question, output_obj, agent_llm, retriever, environment_llm, memory, execute_config):
    agent_type = output_obj['agent']
    if agent_type == AgentType.VALIDATOR.value:
        context = output_obj['input']['information']
        response = output_obj['input']['response']
        output_agent = validate_response(question, context, response, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.SUMMARIZER.value:
        context = output_obj['input']['information']
        output_agent = generate_summary(question, context, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.PLANNER.value:
        context = output_obj['input']['information']
        output_agent = generate_plan(question, context, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.REASONER.value:
        context = output_obj['input']['information']
        aspect = output_obj['input']['aspect']
        output_agent = generate_analysis(question, context, aspect, memory, agent_llm, execute_config)
        return json.dumps(output_agent)
    #elif agent_type == AgentType.COMPLEX_SEARCHER.value:
    #    context = output_obj['input']['information']
    #    output_agent = generate_subquestions(question, context, memory, agent_llm, execute_config)
    #    return json.dumps(output_agent)
    elif agent_type == AgentType.COMPLEX_SEARCHER.value:
        context = output_obj['input']['information']
        # 1) Ask the complex_searcher to (maybe) split the question
        complex_searcher_result = generate_subquestions(question, context, memory, agent_llm, execute_config)
        sub_questions = complex_searcher_result.get("sub_questions", [])
        should_split = complex_searcher_result.get("should_split", True)

        # 2) If complex_searcher chooses not to split, just fall back to one search on the original question
        if (not should_split) or not sub_questions:
            search_result = search(
                question,                 # original question
                context,
                suggestions=[],           # or something simple
                memory=memory,
                llm=agent_llm,
                retriever=retriever,
                execute_config=execute_config,
            )
            # Let the coordinator know what happened
            return json.dumps({
                "mode": "single",
                "used_question": question,
                "found_information": search_result["found_information"],
                "num_documents": len(search_result["documents"]),
            })

        # 3) Otherwise, loop over subquestions and call search() for each
        all_docs_before = len(memory.get("searcher", {}).get("verified_documents", []))
        for sq in sub_questions:
            sq_text = sq["sub_question"]
            # You can use sq["reason"] as suggestions, or leave suggestions empty
            suggestions = [sq.get("reason", "")] if sq.get("reason") else []

            search(
                sq_text,
                context="",               # or keep passing global context if you prefer
                suggestions=suggestions,
                memory=memory,
                llm=agent_llm,
                retriever=retriever,
                execute_config=execute_config,
            )

        all_docs_after = len(memory.get("searcher", {}).get("verified_documents", []))
        num_new_docs = all_docs_after - all_docs_before

        # 4) Return a compact JSON summary so the coordinator sees this as the COMPLEX_SEARCHER's "output"
        return json.dumps({
                "mode": "multi",
                "num_sub_questions": len(sub_questions),
                "sub_questions": sub_questions,
                "num_new_documents": num_new_docs,
                "total_documents": all_docs_after,
            })
    elif agent_type == AgentType.ANSWERER.value:
        guidance = output_obj['input']['guidance']
        important_information = output_obj['input']['important_information']
        verified_documents = memory['searcher']['verified_documents'] if 'searcher' in memory else []
        output_agent = generate_answer(question, verified_documents, guidance, important_information, memory, environment_llm, execute_config)
        memory['responses'].append(output_agent['response'])
        memory['final_response'] = output_agent['response']
        return json.dumps(output_agent)
    elif agent_type == AgentType.REVISER.value:
        suggestion = output_obj['input']['suggestion']
        output_agent = revise_answer(suggestion, memory, environment_llm, execute_config)
        memory['responses'].append(output_agent['response'])
        memory['final_response'] = output_agent['response']
        return json.dumps(output_agent)
    elif agent_type == AgentType.SEARCHER.value:
        context = output_obj['input']['information']
        suggestions = output_obj['input']['suggestions']
        output_agent = search(question, context, suggestions, memory, agent_llm, retriever, execute_config)
        return json.dumps(output_agent)
    elif agent_type == AgentType.FINISHER.value:
        return None
    else:
        raise ValueError("Invalid agent type")

def generate_response(question, execute_config):
    if execute_config['environment_model_server']:
        environment_llm_url = load_url_from_log_file(execute_config['environment_model_server_log_file'])
        environment_llm = SeverLLM(environment_llm_url, execute_config['environment_model'])
    else:
        environment_llm = LLM(execute_config['environment_model'], download_dir=execute_config["download_path"], gpu_memory_utilization=0.45)
    if execute_config['agent_model_server']:
        agent_llm_url = load_url_from_log_file(execute_config['agent_model_server_log_file'])
        jlog(TRACE, "url check", agent_url_passing=agent_llm_url)
        agent_llm = SeverLLM(agent_llm_url, execute_config['agent_model'])
    else:
        agent_llm = LLM(execute_config['agent_model'], download_dir=execute_config["download_path"], gpu_memory_utilization=0.45)
    if execute_config['retriever'] == "BM25":
        retriever = BM25Retriever(execute_config['index_addr'])
    elif execute_config['retriever'] == "lion_sp_llama3_1b":
        retriever = SparseRetriever(execute_config)
    else:
        raise ValueError("Invalid retriever")
    sampling_params = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_agent'], logprobs=1)
    query = question
    
    
    #CHECK FOR EXISTENCE OF EXPERIENCE FILE
    '''
    conversation = None
    experience_path = "/content/drive/MyDrive/mRAG_and_MSRS_source/experiences/experience_base.json_0" 
    if os.path.exists(experience_path):
        conversation = initilize_conversation_informed(question, execute_config['concise'],experience_path)
        jlog(TRACE, "with_routes_passed", agent_url_passing=json.dumps(conversation))
    else:
        conversation = initilize_conversation(query, execute_config['concise'])
    '''
    #conversation = initilize_conversation(query, execute_config['concise'])
    conversation = initilize_conversation_complex(query,True)
        
        
        
    memory = initilize_memory()
    counter = 0
    while counter < execute_config['max_actions']:
        if execute_config['agent_model_server']:
            conversation_text = conversation
        else:
            conversation_text = agent_llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        output_raw = agent_llm.generate(conversation_text, sampling_params)
        output_text = output_raw[0].outputs[0].text
        output_obj = str_to_json(output_text)
        # print(output_obj)
        conversation = update_conversation(conversation, "assistant", output_text)
        user_action = get_next_user_prompt(query, output_obj, agent_llm, retriever, environment_llm, memory, execute_config)
        # print(user_action)
        if user_action is None:
            break
        conversation = update_conversation(conversation, "user", user_action)
        counter += 1
    output_object = {
        "question": query,
        "agent_conversation_text": agent_llm.get_tokenizer().apply_chat_template(conversation, tokenize=False) if not execute_config['environment_model_server'] else conversation,
        "agent_conversation": conversation,
        "verified_documents": memory['searcher']['verified_documents'] if 'searcher' in memory else [],
        "response": memory['final_response'],
        "memory": memory
    }
    return output_object
