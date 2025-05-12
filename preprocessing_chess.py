import fire
import os
# import chromadb
import torch
from tqdm import tqdm
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM
import time
# from deepspeed.ops.op_builder import AsyncIOBuilder
# from deepspeed.ops.op_builder import GDSBuilder
import pathlib

# 4-bit quantization 설정
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # 4-bit로 로드
#     bnb_4bit_compute_dtype=torch.float16,  # 연산 시 float16 사용
#     bnb_4bit_use_double_quant=True,  # 더블 양자화 사용 (메모리 최적화)
#     bnb_4bit_quant_type="nf4"  # NormalFloat4 (nf4) 양자화 적용
# )

def file_write(out_f, tensor, handle, gpu_buffer):
    gpu_buffer.copy_(tensor)
    handle.sync_pwrite(gpu_buffer, out_f)

class DocumentChunk():
  def __init__(
    self,
    id: str,
    text: str, 
  ):
    self.id = id
    self.text = text

class DocumentPreprocessor():
  def __init__(
      self, 
      docs_dir: str,
      cache_dir: str,
      model_name: str = "meta-llama/Llama-3.1-8B", 
  ):
    self.docs_dir = docs_dir
    self.cache_dir = cache_dir
    #model_name = "meta-llama/Llama-2-7b-hf"
    print("Load model")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    self.tokenizer.padding_side = "left"
    self.model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      torch_dtype=torch.float16,
      # quantization_config=bnb_config,
      device_map="cuda",
    )

    print("Model loaded")

  def process_documents(self):
    start_time = time.time()
    files = [f for f in os.listdir(self.docs_dir) if not f.startswith("cache")]
    print(f"Processing {len(files)} documents...")

    for filename in tqdm(files):
      file_path = os.path.join(self.docs_dir, filename)
      pu_name = filename.split('.')[0]
      print(pu_name)
      with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()
        self.save_kv_cache(template, pu_name) # save_kv_cache_aio

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Processing completed in {elapsed_time:.2f} seconds.")

  def save_kv_cache(self, template:str, pu_name:str):
    output_file = os.path.join(self.cache_dir, f"{pu_name}.pt")
    if pu_name in ['IR', 'CG_a', 'CG_b']:
      template = template + "\n"
      input = self.tokenizer(template, return_tensors="pt", padding_side="left").to("cuda")
    else:
      input = self.tokenizer(template, return_tensors="pt", padding_side="left", add_special_tokens=False).to("cuda")
      
    with torch.no_grad():
      output = self.model(**input, use_cache = True)

    cache = output.past_key_values.to_legacy_cache()
    torch.save(cache, output_file)
    '''
      cache = torch.load(os.path.join(self.cache_dir, f"{chunk.id}.pt"))
      past_kv_cache = DynamicCache.from_legacy_cache(loaded)
      self.model(**input, use_cache = True, past_kv_cache = past_kv_cache)
    '''
  
  def save_kv_cache_aio(self, template: str, pu_name:str):
    output_file = os.path.join(self.cache_dir, f"{pu_name}.pt")
    input = self.tokenizer(template, return_tensors="pt", padding_side="left").to("cuda")
    with torch.no_grad():
      output = self.model(**input, use_cache = True)
    cache = output.past_key_values.to_legacy_cache()

    cache_tensors = [t.flatten() for layer in cache for t in layer]  # 하나의 1D 텐서로 변환; .pin_memory()하려면 cpu로 보내야함?
    cache_tensor = torch.cat(cache_tensors)

    aio_handle = AsyncIOBuilder().load().aio_handle()
    bounce_buffer = torch.empty(cache_tensor.shape[0], dtype=torch.float16).pin_memory()

    file_write(output_file, cache_tensor, aio_handle, bounce_buffer)
  
def main(
  docs_dir: str,
  cache_dir: str,
  model_name: str = "meta-llama/Llama-3.1-8B",
):
  preprocessor = DocumentPreprocessor(
    docs_dir=docs_dir,
    cache_dir=cache_dir,
    model_name=model_name
  )

  preprocessor.process_documents()

if __name__ == "__main__":
  fire.Fire(main)
