from typing import Any, Dict, List

from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser

from llm.engine_configs import ENGINE_CONFIGS
from runner.logger import Logger
from threading_utils import ordered_concurrent_function_calls

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers import BitsAndBytesConfig

##### MATKV #####
import json
import os
from transformers import DynamicCache

query_template = '''
【Question】
Question: 
{question}

Evidence:
{evidence}query

**************************
【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer.
'''

base_dir = os.getcwd()

def load_kv_cache(pu_name: str):
    if pu_name in ['CG_a', 'CG_b']:
        cache_dir = f'{base_dir}/chess_pu/agent_template/cache_8b/'
    else:
        cache_dir = f'{base_dir}/chess_pu/db_schema/cache_8b/'
    in_file = os.path.join(cache_dir, f"{pu_name}.pt")
    
    return torch.load(in_file, weights_only=True, map_location="cuda")

def load_all_caches(pus: List[str]):
    return [load_kv_cache(pu) for pu in pus]

def concat_caches_single(caches):
    num_layers = len(caches[0])
    concatenated = []
    for layer in range(num_layers):
        keys = torch.cat([cache[layer][0] for cache in caches], dim=2)
        values = torch.cat([cache[layer][1] for cache in caches], dim=2)
        concatenated.append((keys, values))
    return concatenated

def create_pu_caches(pu_list: List[str]):
    # load_all_caches: agent_template, db_schema seperately
    caches = load_all_caches(pu_list)

    if len(caches) == 1:
        past_kv_caches = DynamicCache.from_legacy_cache(caches)
    else:
        past_kv_caches = DynamicCache.from_legacy_cache(concat_caches_single(caches))
    return past_kv_caches

def load_prompt(pu_list: List[str]):
    prompt = ""
    for pu_name in pu_list:
        if pu_name in ['CG_a', 'CG_b']:
            cache_dir = f'{base_dir}/chess_pu/agent_template/'
            in_file = os.path.join(cache_dir, f"{pu_name}.txt")
            with open(in_file, 'r', encoding='utf-8') as f:
                content = f.read()
                content = content + '\n'
        else:
            cache_dir = f'{base_dir}/chess_pu/db_schema/'
            in_file = os.path.join(cache_dir, f"{pu_name}.txt")
            with open(in_file, 'r', encoding='utf-8') as f:
                content = f.read()
        prompt += content
    return prompt
##### MATKV #####


_LLAMA_MODEL = None
_LLAMA_TOKENIZER = None

def _init_llama_once(model_name="meta-llama/Llama-3.1-8B"):
# def _init_llama_once(model_name="meta-llama/Meta-Llama-3-8B"):
    global _LLAMA_MODEL, _LLAMA_TOKENIZER
    if _LLAMA_MODEL is None:
        _LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        _LLAMA_TOKENIZER.pad_token = _LLAMA_TOKENIZER.eos_token
        
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     llm_int8_enable_fp32_cpu_offload=True,    
        # )
        # _LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     quantization_config=bnb_config,
        #     device_map="auto",
        # ) #FIXME
        _LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        _LLAMA_MODEL.eval()
    return _LLAMA_MODEL, _LLAMA_TOKENIZER

    
def get_llm_chain(engine_name: str, temperature: float = 0, base_uri: str = None) -> Any:
    """
    Returns the appropriate LLM chain based on the provided engine name and temperature.

    Args:
        engine (str): The name of the engine.
        temperature (float): The temperature for the LLM.
        base_uri (str, optional): The base URI for the engine. Defaults to None.

    Returns:
        Any: The LLM chain instance.

    Raises:
        ValueError: If the engine is not supported.
    """
    if engine_name not in ENGINE_CONFIGS:
        raise ValueError(f"Engine {engine_name} not supported")
    
    config = ENGINE_CONFIGS[engine_name]
    constructor = config["constructor"]
    params = config["params"]
    if temperature:
        params["temperature"] = temperature
    
    # Adjust base_uri if provided
    if base_uri and "openai_api_base" in params:
        params["openai_api_base"] = f"{base_uri}/v1"
    
    model = constructor(**params)
    if "preprocess" in config:
        llm_chain = config["preprocess"] | model
    else:
        llm_chain = model
    return llm_chain


def call_llama(
    prompt: Any, 
    engine: Any, parser: Any, request_kwargs: Dict[str, Any], step: int, 
    template_name: str, 
    max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60):

    logger = Logger()
    model, tokenizer = _init_llama_once()
    
    # print("REQUEST:", request_kwargs)
    
    if template_name == "generate_candidate_one":
        pu_list = ['CG_a', 'california_schools']
        past_key_values = create_pu_caches(pu_list)
        prompt = load_prompt(pu_list)

    elif template_name == "generate_candidate_two":
        pu_list = ['CG_b', 'california_schools']
        past_key_values = create_pu_caches(pu_list)
        prompt = load_prompt(pu_list)
    
    if past_key_values:
        past_key_values_clone = [
            (k.clone().detach(), v.clone().detach()) for k, v in past_key_values
            ]
        # print("past_key_values_shape:", past_key_values_clone[0][0].shape)
    
    # max_attempts = 5 # FIXME (attempts는 에러가 발생하면 다음 차례로 seq. 처리 되어서 KV Cache 누적 안 되어 OOM 발생에 영향을 미치지 않는 것으로 판단 됨. OOM 문제는 candidate 여러 개 만들 때, worker 나뉘는 문제.)
    
    query = query_template.format(
        question=request_kwargs["QUESTION"],
        evidence=request_kwargs["HINT"],
    )
    
    prompt_text = prompt + query
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_token_length = inputs.input_ids.shape[1]
    # print("prompt_token_length:", prompt_token_length)
            
    for attempt in range(max_attempts):
        try:
            if past_key_values:
                kv_cache = tuple(past_key_values_clone)
            else:
                kv_cache = None
        
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    use_cache = True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    return_legacy_cache=True,
                    temperature=0.5,
                    past_key_values=None,
                )

            # prompt 제외 output만 decode
            generated_text = outputs.sequences[0][prompt_token_length:]
            answer = tokenizer.decode(generated_text, skip_special_tokens=True).strip()
            parsed = parser.invoke(answer)
            
            # 미리 생성된 sql 확인
            if "</FINAL_ANSWER>" in answer:
                print(answer.split("<FINAL_ANSWER>")[1].split("</FINAL_ANSWER>")[0])
            else:
                print("error (need more tokens)")
            # print("PARSED:", parsed)
            return parsed

        except Exception as e:
            logger.log(f"Failed attempt {attempt+1}: {e}", "error")
            if attempt == max_attempts - 1:
                raise

                
def async_llama_chain_call(
    prompt: Any,
    engine: Any, 
    parser: Any, 
    request_list: List[Dict[str, Any]], 
    step: int, 
    template_name: str,
    sampling_count: int = 1,
) -> List[List[Any]]:

    call_list = []
    engine_id = 0
    
    # if matkv:
    #     past_key_values_clone = [
    #     (k.clone().detach(), v.clone().detach()) for k, v in matkv
    #     ]
    
    for request_id, request_kwargs in enumerate(request_list):
        for _ in range(sampling_count):
            call_list.append({
                'function': call_llama,
                'kwargs': {
                    'prompt': prompt,
                    'engine': engine[engine_id % len(engine)] if isinstance(engine,list) else engine,
                    'parser': parser,
                    'request_kwargs': request_kwargs,
                    'step': step,
                    'template_name': template_name,
                }
            })
            engine_id += 1

    # Execute the functions concurrently
    results = ordered_concurrent_function_calls(call_list)

    # Group results by sampling_count
    grouped_results = [
        results[i * sampling_count: (i + 1) * sampling_count]
        for i in range(len(request_list))
    ]

    return grouped_results

def call_llm_chain(prompt: Any, engine: Any, parser: Any, request_kwargs: Dict[str, Any], step: int, max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
        max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
        backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The output from the chain.

    Raises:
        Exception: If all attempts fail.
    """
    logger = Logger()
    for attempt in range(max_attempts):
        try:
            # chain = prompt | engine | parser
            chain = prompt | engine
            prompt_text = prompt.invoke(request_kwargs).messages[0].content
            # print(prompt_text)
            output = chain.invoke(request_kwargs)
            # print("OUTPUT")
            # print(output)
            if isinstance(output, str):
                if output.strip() == "":
                    engine = get_llm_chain("gemini-1.5-flash")
                    raise OutputParserException("Empty output")
            else:
                if output.content.strip() == "":    
                    engine = get_llm_chain("gemini-1.5-flash")
                    raise OutputParserException("Empty output")
            output = parser.invoke(output)
            logger.log_conversation(
                [
                    {
                        "text": prompt_text,
                        "from": "Human",
                        "step": step
                    },
                    {
                        "text": output,
                        "from": "AI",
                        "step": step
                    }
                ]
            )
            return output
        except OutputParserException as e:
            logger.log(f"OutputParserException: {e}", "warning")
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=engine)
            chain = prompt | engine | new_parser
            if attempt == max_attempts - 1:
                logger.log(f"call_chain: {e}", "error")
                raise e
        except Exception as e:
            # if attempt < max_attempts - 1:
            #     logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
            #     sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
            #     time.sleep(sleep_time)
            # else:
            logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error")
            raise e


def async_llm_chain_call(
    prompt: Any, 
    engine: Any, 
    parser: Any, 
    request_list: List[Dict[str, Any]], 
    step: int, 
    sampling_count: int = 1
) -> List[List[Any]]:
    """
    Asynchronously calls the LLM chain using multiple threads.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_list (List[Dict[str, Any]]): The list of request arguments.
        step (int): The current step in the process.
        sampling_count (int): The number of samples to be taken.

    Returns:
        List[List[Any]]: A list of lists containing the results for each request.
    """

    call_list = []
    engine_id = 0
    for request_id, request_kwargs in enumerate(request_list):
        for _ in range(sampling_count):
            call_list.append({
                'function': call_llm_chain,
                'kwargs': {
                    'prompt': prompt,
                    'engine': engine[engine_id % len(engine)] if isinstance(engine,list) else engine,
                    'parser': parser,
                    'request_kwargs': request_kwargs,
                    'step': step
                }
            })
            engine_id += 1

    # Execute the functions concurrently
    results = ordered_concurrent_function_calls(call_list)

    # Group results by sampling_count
    grouped_results = [
        results[i * sampling_count: (i + 1) * sampling_count]
        for i in range(len(request_list))
    ]

    return grouped_results


def call_engine(message: str, engine: Any, max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        message (str): The message to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
        backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The output from the chain.

    Raises:
        Exception: If all attempts fail.
    """
    logger = Logger()
    for attempt in range(max_attempts):
        try:
            output = engine.invoke(message)
            return output.content
        except Exception as e:
            # if attempt < max_attempts - 1:
            #     logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
            #     sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
            #     time.sleep(sleep_time)
            # else:
            logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error")
            raise e
