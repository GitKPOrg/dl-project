'''
Design decisions:
- Type of model: Instruct vs base model: first already trained to follow instructions
- Model: llama-8B more capable than llama-3B but difficult to finetune in colab (RAM, GPU restrictions); chose llama-3B 
- BitsAndBytes: Using 4-bit quantized form reduces RAM requirements (from 32–40 down to 8–12 GB) by storing weights in 4-bit and computing in float16; necessary to handle model on colab
- LoRA: Reduces number of trainable parametres significantly; neccesary for efficient colab finetuning
- Predict: Implemented without batching for now 
'''

# libraries
# anaconda prompt: pip install openai
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# common methods
class CommonMethods:
  '''
  Purpose: Create some methods that the llama3 models need
  Class attributes: NA
  Instance attributes: classification options and mappings
  '''
    
  def __init__(self):
    self.answer_options = ["negative", "neutral", "positive"] 
    self.mapping_txt_to_num = {"negative": 0, "neutral": 1, "positive":2, None:None} 
    self.mapping_num_to_txt = {0: "negative", 1: "neutral", 2: "positive", None: None}

  def label3_txt_to_num(self, ls):
    '''
    Purpose: maps list of "negative", "neutral", "positive" labels to list of 
    0, 1, or 2 respectively (and None to None, see dict mapping)
    '''
    return [self.mapping_txt_to_num[i] for i in ls] 

  def label3_num_to_txt(self, ls):
    '''
    Purpose: maps list of 0, 1, 2 labels to list of negative", "neutral", "positive"
    (and None to None, see dict mapping)
    '''
    return [self.mapping_num_to_txt[i] for i in ls] 

  def generate_prompt(self, instruction, text, label=None):
    '''
    Generares a prompt with instruction, text, and label. For training, include the label;
    for val or test, don't include the label (i.e. set label=None)
    Note: Label must be str not num.
    '''
    if label is None:
      label = ""

    prompt = f"""
    Instruction: {instruction}
    Text: {text}
    Label: {label}
    """.strip()

    return prompt
    
  def predict(self, model, tokenizer, instruction, x, text_token_limit=256, report_interval=100):
    '''
    Predicts a llama3 model
    Inputs:
    - x array with text data to be classified (e.g., a pandas data series),
    - model: a llama3 model (default or finetuned)
    - tokenizer: a llma3 tokenizer
    - text_token_limit: int; number of tokens to truncate the text
    - report_interval=int; log prediction process at every teport_interval
    Outputs:
    - y_hat: list with predicted labels
    TODO: Works as is but could profit from batching
    '''

    y_hat = []
    
    for index, text in enumerate(x):

      # truncate text at token max:
      # Use instead of max_length parameter in tokenizer to avoid cutting off trailing "label:"
      ids = tokenizer(text, add_special_tokens=False)["input_ids"]
      if len(ids) > text_token_limit:
        ids = ids[:text_token_limit]
        text = tokenizer.decode(ids, skip_special_tokens=True)

      # create prompt
      prompt = self.generate_prompt(instruction=instruction, text=text, label=None)

      # tokenize
      device = next(model.parameters()).device
      inputs = tokenizer(
          prompt,
          return_tensors="pt").to(device)

      # generate output
      with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False)

      # process output (isolate generated output, ids to token, lower, remove punctuation, filter)
      output_generated = output[0][inputs["input_ids"].shape[1]:] # slice from tot output (prompt + gen. output) the gen. output
      output_generated = tokenizer.decode(output_generated, skip_special_tokens=True)
      output_generated = output_generated.lower().strip(".,:;!?\"'")
      label = "neutral" # instead of neutral sometimes ? etc returned
      for option in self.answer_options:
        if option in output_generated:
          label = option
          break

      # save prediction
      y_hat.append(label)

      # log progess
      if report_interval and (index + 1) % report_interval == 0:
        print(f"Reviews processed: {index + 1}")

    return y_hat

class LlamaBaseModel:
  '''
  Purpose: Create a base llama model without lora. Intended use is zeroshot or fewshot, i.e. 
  predicting without finetuning (finetunig this model would be possible but require 
  a lot of GPU and RAM).
    
  Class attributes:
  - model_name: str; tested for: "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"
    but other options might work as well
  - quant config: dict; BitsAndBytes configuration (see deisgn note at beginning)
  - instruction: str; tells the model what to do (analogous to developer message in OpenAI)

  Methods:
  - predict  
  '''
  def __init__(self, model_name, quant_config, instruction):
    self.common_methods = CommonMethods()
    self.model_name = model_name
    self.quant_config = quant_config
    self.instruction = instruction

    # create model
    self.model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=model_name,
      quantization_config=quant_config,
      device_map="auto",
      trust_remote_code=True)
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(device)
    # self.model.device = device # ensure model exposes a .device attribute for accelerate

    # create corresponding tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def predict(self, x, text_token_limit=256, report_interval=100):
    '''
    Wrapper for CommonMethods.predict 
    '''
    return self.common_methods.predict(
      model=self.model,
      tokenizer=self.tokenizer,
      instruction=self.instruction,
      x=x,
      text_token_limit=text_token_limit,
      report_interval=report_interval)
  
class LlamaLoraModel:
  '''
  Purpose: Create a base llama model with lora. Intended use is finetuning and 
  then predicting.
    
  Class attributes:
  - model_name: str; tested for: "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"
    but other options might work as well
  - quant config: dict; BitsAndBytes configuration (see deisgn note at beginning)
  - instruction: str; tells the model what to do (analogous to developer message in OpenAI)

  Methods:
  - train
  - predict
  '''
  def __init__(self, model_name, quant_config, peft_config, instruction):
    self.common_methods = CommonMethods()
    self.model_name = model_name
    self.quant_config = quant_config
    self.peft_config = peft_config
    self.instruction = instruction

    # create model
    self.model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=model_name,
      quantization_config=quant_config,
      device_map="auto",
      trust_remote_code=True)

    # apply lora to model 
    self.model = get_peft_model(self.model, self.peft_config)
    self.model.print_trainable_parameters()

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(device)
    # self.model.device = device # ensure model exposes a .device attribute for accelerate

    # create correspnding tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def train(self, x_train, y_train, x_val, y_val, training_config, text_token_limit=256):
    '''
    Purpose: Predict a llama3 model
    Inputs:
    - x_train: x array with text to be classified (e.g., a pandas data series)   
    - y_train: x array with the label as string (e.g., a pandas data series) 
    - x_val, y_val: analogus
    - text_token_limit: int; number of tokens to truncate the text
    - training_config: object that specifies training arguments
    Outputs:
    - self.model is updated during trainng/ fine-tuning
    - self.train_result: final training output summary (overall metrics, final loss, steps, runtime
    - self.log_history: full chronological log of all training/eval events
    '''

    # create prompt data from x and y data  
    prompt_train = [
      self.common_methods.generate_prompt(self.instruction, text, label) 
      for text, label in zip(x_train, y_train)]

    prompt_val = [
      self.common_methods.generate_prompt(self.instruction, text, label) 
      for text, label in zip(x_val, y_val)]

    ds_prompt_train = Dataset.from_dict({"prompt": prompt_train})
    ds_prompt_val = Dataset.from_dict({"prompt": prompt_val})

    # Create encoded prompts: performs tokenization, builds attention masks, assigns labels
    # Note: resulting encoding dict contains: input_ids, attention_mask, labels
    def encode(batch):
        encoding = self.tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=text_token_limit,
            padding="max_length")
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    ds_prompt_train_enc = ds_prompt_train.map(encode, batched=True, remove_columns=["prompt"])
    ds_prompt_train_enc.set_format(type="torch")

    ds_prompt_val_enc = ds_prompt_val.map(encode, batched=True, remove_columns=["prompt"])
    ds_prompt_val_enc.set_format(type="torch")

    # training
    self.trainer = Trainer(
        model=self.model,
        args=training_config,
        train_dataset=ds_prompt_train_enc,
        eval_dataset=ds_prompt_val_enc)

    self.train_result = self.trainer.train()
    self.log_history = self.trainer.state.log_history

  def predict(self, x, text_token_limit=256, report_interval=100):
    '''
    Wrapper for CommonMethods.predict 
    '''
    return self.common_methods.predict(
      model=self.model,
      tokenizer=self.tokenizer,
      instruction=self.instruction,
      x=x,
      text_token_limit=text_token_limit,
      report_interval=report_interval)
