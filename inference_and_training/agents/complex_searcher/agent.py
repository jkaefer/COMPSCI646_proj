from vllm import LLM, SamplingParams
from utils.json_utils import str_to_json



COMPLEX_SEARCHER_SYSTEM_PROMPT = """You are a helpful and capable agent whose task is to
decompose a complex question into several simpler sub-questions that, taken
together, allow answering the original question.

# Your input:
    - "question": the original question the user wants to answer.
    - "information": a summary of the information the user has gathered so far.
      This can be empty if the user has not gathered any information yet.

# Your output: you need to provide a JSON object enclosed in ```json ``` that contains:
    - "should_split": a boolean. If false, you think the original question should be treated as a single question.
    - "sub_questions": a list of objects. Each object has:
        - "id": integer
        - "sub_question": string
        - "reason": string

Your output must be a valid JSON object enclosed in ```json ``` with exactly the
field "sub_questions" as described, and no extra commentary before or after.
"""



COMPLEX_SEARCHER_USER_PROMPT = """# question: {QUESTION}
# information: {INFORMATION}
"""

def initilize_conversation():
    conversation = [
        {
            "role": "system",
            "content": COMPLEX_SEARCHER_SYSTEM_PROMPT
        }
    ]
    return conversation




def generate_subquestions(question, information, memory, llm, execute_config):
    # Store complex_searcher conversation in memory just like other agents
    if 'complex_searcher' not in memory:
        memory['complex_searcher'] = initilize_conversation()
        memory['complex_searcher_state'] = {"sub_questions": []}

    conversation = memory['complex_searcher']
    conversation.append({
        "role": "user",
        "content": COMPLEX_SEARCHER_USER_PROMPT.format(
            QUESTION=question,
            INFORMATION=information
        )
    })


    if execute_config['agent_model_server']:
        conversation_text = conversation
    else:
        conversation_text = llm.get_tokenizer().apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        
    sampling_parmas = SamplingParams(temperature=execute_config['temperature_agent'], top_p=execute_config['top_p'], max_tokens=execute_config['max_tokens_environment'], logprobs=1)

    response_text = llm.generate(conversation_text, sampling_parmas)[0].outputs[0].text
    response_obj = str_to_json(response_text)


    conversation.append({
        "role": "assistant",
        "content": response_text
    })

    sub_questions = response_obj.get("sub_questions", [])
    memory['complex_searcher_state']['sub_questions'] = sub_questions

    return {
        "sub_questions": sub_questions
    }

