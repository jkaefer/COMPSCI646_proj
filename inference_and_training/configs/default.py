DEFAULT_CONFIG = {
    "download_path": "/content/drive/MyDrive/mRAG_and_MSRS_source/vllm_cache",
    "agent_model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "environment_model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "environment_model_server": True,
    "environment_model_server_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/agent.log",
    "agent_model_server": True,
    "agent_model_server_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/agent.log",
    "retriever": "BM25",
    "retriever_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/retriever_log/retriever.log",
    "index_addr": "/content/drive/MyDrive/mRAG_and_MSRS_source/index",
    "max_actions": 20,
    "max_tokens_agent": 32000,
    "max_tokens_environment": 8192,
    "temperature_agent": 0.1,
    "temperature_environment": 0.1,
    "top_p": 0.95,
    "top_k":10,
    "threshold":0.0,
    "max_verifcation_same_query": 5,
    "max_retries": 20,
    "concise": True,
}

DEFAULT_CONFIG_2 = {
    "download_path": "/content/drive/MyDrive/mRAG_and_MSRS_source/vllm_cache",
    "agent_model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "environment_model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "environment_model_server": True,
    "environment_model_server_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/agent.log",
    "agent_model_server": True,
    "agent_model_server_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/agent_server_log/agent.log",
    "retriever": "BM25",
    "retriever_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/retriever_log/retriever.log",
    "index_addr": "/content/drive/MyDrive/mRAG_and_MSRS_source/index",
    "max_actions": 20,
    "max_tokens_agent": 32000,
    "max_tokens_environment": 8192,
    "temperature_agent": 0.1,
    "temperature_environment": 0.1,
    "top_p": 0.95,
    "top_k":10,
    "threshold":0.0,
    "max_verifcation_same_query": 5,
    "max_retries": 20,
    "concise": True
}

DEFAULT_CONFIG_EVALUATION = {
    "download_path": "/content/drive/MyDrive/mRAG_and_MSRS_source/vllm_cache",
    "judge_model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "judge_model_server": True,
    "judge_model_server_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/judge_server_log/judge.log",
    "max_tokens_judge": 16000,
    "temperature_judge": 0.5,
    "top_p": 0.95,
    "num_samples_judge": 5,
    "ragas": True
}

DEFAULT_CONFIG_EVALUATION_2 = {
    "download_path": "/content/drive/MyDrive/mRAG_and_MSRS_source/vllm_cache",
    "judge_model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "judge_model_server": True,
    "judge_model_server_log_file": "/content/drive/MyDrive/mRAG_and_MSRS_source/judge_server_log/judge.log",
    "max_tokens_judge": 16000,
    "temperature_judge": 0.5,
    "top_p": 0.95,
    "num_samples_judge": 5,
    "ragas": True
}
