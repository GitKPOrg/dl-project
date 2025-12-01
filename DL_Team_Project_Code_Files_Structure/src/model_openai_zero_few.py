'''
Design decisions:
- structured output: recom.; goes together with respnose.parse() instead response.creat()
- output format: recom. to let model response with strings not ingegers; use "negative", 
  "positive", "neutral", not integers 
- prompting advice: use message roles and instructions (developer and user role). 
  For developer message: use structure: identity, instructions, examples (only for few shot), 
  context (na here); use markup and xml tags; prompt cache does not play a role as prompt to short  
- batching vs individual: recom. only one example per call for base version, 
  so individual; this means to loop through all examples inserting it into prompt.
  Stretch: try micro batching in the prompt with list schema 
- model: gpt-5-nano is the cheapest; gpt-4.1-mini is supposed to be well suited for this task. 
  But use other model for comparison in case of fine-tuning
- Error handling: in case of error, return an None or Na (as result) and the 
  print the error message to the screen 
- style: object oriented
'''

# libraries
# anaconda prompt: pip install openai
from openai import OpenAI, OpenAIError 
import pandas as pd
from pydantic import BaseModel
from typing import Literal

# model class
class OpenAiModelZeroFew:
    '''
    Purpose: Use OpenAI API for zero-shot and few-shot sentiment analysis of 
    consumer reviews. Note: Few-shot differs from zero-shot only in the examples
    that are added in the developer-message.
    
    Class attributes:
    - max_output_tokens: limited to 16 (min) to save cost if things go wring
    
    Instance attributes:
    - model: model
    - developer_message: str containing the message of the developer. Note: For
      few-shot include examples.  
    
    Predit method inputs:
    - reviews: pandas series containing the reviews (txt)
    - report_interval: log reviews processed; none (default) or int 

    Output:
    - list with predicted labels

    Notes:
    - to get costs (i.e., number of tokens): after model.predict() use 
      model.input_tokens() and output_tokens()
    '''
    # max_output_tokens = 16 # buggy with gpt-5-nano

    # strucutred output
    class Label(BaseModel):
        label: Literal["negative", "neutral", "positive"] 

    def __init__(self, model, developer_message):
        self.model = model
        self.developer_message = developer_message
        self.client = OpenAI()

        # mapping label3 text to int
        self.mapping = {"negative": 0, "neutral": 1, "positive":2, None:None} 

    def predict(self, reviews, report_interval=None):
        '''
        Purpose: does the sentimen analysis       
        '''   

        # loop using response API
        pred_txt = []
        self.input_tokens = 0
        self.output_tokens = 0

        for index, review in enumerate(reviews):
            
            try: 
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text", 
                                    "text": self.developer_message,
                                }]
                        },
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "input_text", 
                                    "text": review,
                                }]
                        },
                    ],
                    text_format=self.Label,
                    # temperature=0, # not supported for gpt-5-nano
                    # max_output_tokens=max_output_tokens # buggy for gpt-5-nano 
                )
                # add response to output list
                pred_txt.append(response.output_parsed.label)
            
                # log cost
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens

            except OpenAIError as e:
                print(f"Error: {e}")
                pred_txt.append(None) 

            # log progess
            if report_interval and (index + 1) % report_interval == 0:
                    print(f"Reviews processed: {index + 1}")

        return pred_txt

    def label3_text_to_num(self, ls):
        '''
        Purpose: maps "negative", "neutral", "positive" labels to integers
        0, 1, or 2 respectively (and None to None, see dict mapping)
        '''
        ls_out = [self.mapping[i] for i in ls] 
        return ls_out