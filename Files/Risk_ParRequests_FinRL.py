import os
import time
import json
import tiktoken
import threading
from pydantic import BaseModel
from threading import Semaphore
from typing import List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
    retry_if_exception_type
)
import yaml
import numpy as np
import pandas as pd
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import time
import json
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random, retry_if_exception_type

import csv
import sys

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

import logging

logging.basicConfig(
    filename="script.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Script started.")


MAX_WORKERS = 300  # Adjust based on your API rate limits
from tenacity import retry, stop_after_attempt, wait_exponential

class ParallelAPIRequesterConfig(BaseModel):
    provider: Literal["deepseek-chat", "openai", "azure"]  # Added DeepSeek
    model_name: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 1.0
    request_rate_limit: Optional[int] = 100  # Default to DeepSeek's free tier limit
    token_rate_limit: Optional[int] = 15000  # Default token limit
    cost_print_interval: Optional[int] = 50
    check_Intent_keys_and_values: Optional[bool] = False
    budget: Optional[float] = 30
    return_in_json_format: Optional[bool] = True

class TokenSemaphore:
    '''
    
    Custom TokenSemaphore for Token Rate Limiting

    The Python standard Semaphore from the threading module starts with an internal counter, which you specify upon creation. This counter decrements each time acquire() is called and increments when release() is called.
    However, the standard Semaphore doesn't support acquiring or releasing more than one unit of the counter at a time, which means it can't directly manage multiple tokens per request out-of-the-box if those requests consume a variable number of tokens.

    The following custom class allows you to specify how many tokens to acquire or release at a time, giving you the flexibility needed for handling variable token counts per API request.
    '''

    def __init__(self, max_tokens):
        self.tokens = max_tokens
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def acquire(self, required_tokens):
        with self.lock:
            while self.tokens < required_tokens:
                self.condition.wait()
            self.tokens -= required_tokens

    def release(self, released_tokens):
        with self.lock:
            self.tokens += released_tokens
            self.condition.notify_all()

#NO_ANSWER_TEXT = "NO ANSWER"
#PROMPT_NO_ANSWER = f"If you cannot provide an answer, answer with `{NO_ANSWER_TEXT}`."
#PATTERN_SEP = rf"\n(.*\n)*?"
PATTERN_ANSWER = rf"\b[1-5]\b" # I need to fix this to be a number INT
PATTERN_FLOAT = rf"\d*\.?\d+"
PROMPT_EXAMPLE_FIVE_SHOT_SCORES = [0.80, 0.43, 0.71, 0.34, 0.08] # redo it as needed

PATTERN_ANSWER = r"\b[1-5]\b"  # Ensures risk scores are integers 1-5
# Update response patterns to match the new format

response_patterns = [
    re.compile(fr"Risk:?\s*(?P<score>{PATTERN_ANSWER})\D+(Probability:?\s*)?(?P<confidence>{PATTERN_FLOAT})", re.IGNORECASE),
    re.compile(fr"Answer:?\s*(?P<score>{PATTERN_ANSWER}).*?Probability:?\s*(?P<confidence>{PATTERN_FLOAT})", re.IGNORECASE),
]

def extract_from_response(response, patterns, names):
    # remove formatting
    response = response.replace("**", "")
    response = response.replace("__", "")
    response.replace("```", "")
    # try each pattern
    for pattern in patterns:
        match = pattern.search(response)
        if match is not None:
            if isinstance(names, str):
                return match.group(names)
            else:
                return (match.group(name) for name in names)
    #print('this is the extract from response == ', match.group(names))
    if isinstance(names, str):
        return None
    else:
        return (None for _ in names)
def extract_answer(response):
    risks, confprobs = [], []
    # Split responses by "Risk:" delimiter
    entries = re.split(r'(?=\bRisk:\s*\d+)', response)
    for entry in entries:
        if not entry.strip():
            continue
        score, confidence = extract_from_response(entry, response_patterns, ("score", "confidence"))
        try:
            risk_value = int(score.strip()) if score else np.nan
            conf_value = float(confidence.strip()) if confidence else np.nan
        except:
            risk_value = np.nan
            conf_value = np.nan
        risks.append(risk_value)
        confprobs.append(conf_value)
    return risks, confprobs

openai = OpenAI( 
                api_key="sk-7c563985e05342108b5f27245c59b167",
                base_url="https://api.deepseek.com"  # Verify actual endpoint
            )
#class ParallelDeepSeekRequester:
    #def __init__(self, name, config: ParallelAPIRequesterConfig):
        #assert config.request_rate_limit is not None, "request_rate_limit required"
        #assert config.token_rate_limit is not None, "token_rate_limit required"

        #self.provider = config.provider
model_name = 'deepseek-chat'
max_tokens = 1000
temperature = 0

        # Rate limits (DeepSeek's free tier: 60 RPM / 15K TPM)
request_rate_limit = 300
request_semaphore = Semaphore(request_rate_limit)
token_rate_limit = 15000
token_semaphore = TokenSemaphore(token_rate_limit)

        # Cost tracking (example prices - verify with DeepSeek's actual pricing)
price_list = {
            "deepseek-chat": {
                "input": 0.00007,  # $0.070 per million tokens
                "output": 0.00007   # $0.070 per million tokens
            }
        }
cost = 0.0
budget = 200

# Rate limit validation
max_request_limit = 60  # Free tier
max_token_limit = 15000
#assert self.request_rate_limit <= max_request_limit, f"DeepSeek limit: {max_request_limit} RPM"
#assert self.token_rate_limit <= max_token_limit, f"DeepSeek limit: {max_token_limit} TPM"
        
        # Rest of initialization remains similar...

    # Token counting logic remains unchanged (verify if DeepSeek uses same tokenizer)
@retry(
    wait=wait_fixed(0) + wait_random(0, 1),
    stop=stop_after_attempt(3),
    #retry=retry_if_exception_type((openai.APIConnectionError, openai.APIError)),
    before_sleep=lambda _: print("Retrying...")
)
def get_response(system_user_message):
        #print(f"Sending request: {system_user_message}")
        try:

            response = openai.chat.completions.create(
            model=model_name,
            messages=system_user_message,
            temperature=0,
            max_tokens=8000,
        )
        # Handle usage data safely
            #if response.usage:
            #    input_token_count += response.usage.prompt_tokens
            #    output_token_count += response.usage.completion_tokens
            #    cost += (
            #    response.usage.prompt_tokens * price_list[model_name]["input"] +
            #    response.usage.completion_tokens * price_list[model_name]["output"]
            #)
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {str(e)}")
            raise
            #""" response = self.client.chat.completions.create(
            #    model=self.model_name,
            #    messages=system_user_message,
            #    temperature=self.temperature,
            #    max_tokens=self.max_tokens,
            #) """
            #openai = OpenAI(api_key = "sk-7c563985e05342108b5f27245c59b167",  # Replace with your actual DeepInfra token
            #base_url = "https://api.deepseek.com")
           
            
            # Check if the response has the expected structure
            if not response or "choices" not in response or not response.choices:
                print(f"Invalid response structure: {response}")
                raise AttributeError("Invalid response structure")

            generated_response = response.choices[0].message.content
            print(f"Generated response: {generated_response}")

            # Update cost tracking
            self.input_token_count += response.usage.prompt_tokens
            self.output_token_count += response.usage.completion_tokens
            self.cost += response.usage.prompt_tokens * self.price_list[self.model_name]["input"] + \
                         response.usage.completion_tokens * self.price_list[self.model_name]["output"]

            return generated_response
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            raise

def get_responses_parallel( messages_list, config):
        results = []
        started = time.time()
        request_counter = 0

        # Ensure each item in the list has the key "api_message"
        #assert all("api_message" in item for item in messages_list), "Each item in the list must have the key api_message"
        identifier_keys = [key for key in messages_list[0] if key != "api_message"]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks to the executor, associating each future with its corresponding index and message
            future_to_info = {executor.submit(get_response, item['api_message']): item for item in messages_list}

            for future in as_completed(future_to_info):
                item = future_to_info[future]
                request_counter += 1

                try:
                    # Get the result from the future
                    response = future.result()
                    if response is None:
                        print(f"No response for {item['symbol']} on {item['date']}.")
                        continue

                    # Debug the response
                    #print(f"Response for {item['symbol']} on {item['date']}: {response}")

                    # Extract risk and probability from the response
                    risks, probabilities = extract_answer(response)
                    risk = risks[0] if risks else None
                    probability = probabilities[0] if probabilities else None

                    # Build the result dictionary
                    result_dict = {key: item[key] for key in identifier_keys}
                    result_dict.update({
                        "Risk": risk,
                        "Probability": probability,
                        "response": response
                    })
                    results.append(result_dict)

                except Exception as e:
                    # Handle errors and include them in the results
                    print(f"Error processing message for {item['symbol']} on {item['date']}: {str(e)}")
                    result_dict = {key: item[key] for key in identifier_keys}
                    result_dict.update({
                        "Risk": None,
                        "Probability": None,
                        "response": f"Error: {str(e)}"
                    })
                    results.append(result_dict)

                # Print progress
                if request_counter % 100 == 0 or request_counter == len(messages_list):
                    print(f"Processed {request_counter}/{len(messages_list)} requests.")

                # Check if the budget has been exceeded
                if cost >= budget:
                    print(f"Budget of {budget} dollars exceeded. Stopping further requests.")
                    break

        # Print total time taken
        elapsed_time = time.time() - started
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total time taken: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

        return results


# start exper with the LLM API    




# Constants
INPUT_FILE = r'/homes/eya24/projects/FinRL_DeepSeek/nasdaq_news_all.csv'
OUTPUT_FILE = r'/homes/eya24/projects/FinRL_DeepSeek/risk_deepseek_Nasdaq_me.csv'
CHUNK_SIZE = 1000
MAX_WORKERS = 300  # Adjust based on your API rate limits

# Prepare requests from the input CSV
chunks = pd.read_csv(INPUT_FILE, encoding="utf-8", chunksize=CHUNK_SIZE, on_bad_lines='warn', engine='python')
prepared_requests = []

for chunk in chunks:
    for index, row in chunk.iterrows():
        # Ensure required columns exist and are not null
        if pd.isna(row.get('Stock_symbol')) or pd.isna(row.get('Date')) or pd.isna(row.get('Lsa_summary')):
            continue  # Skip rows with missing data

        # Append the request with the index for merging later
        prepared_requests.append({
            "index": index,
            "symbol": row['Stock_symbol'],
            "date": row['Date'],
            "lsa_summary": row['Lsa_summary'],
            "api_message": [{"role": "system", "content": f"You are a financial risk analyst assessing stocks as of {row['Date']}. Evaluate news for {row['Stock_symbol']} and assign: \n- A risk score (1-5): 1=very low, 2=low, 3=moderate (default if unclear), 4=high, 5=very high. \n- A probability (0.00-1.00) reflecting confidence, based on news clarity, data availability, and task difficulty. \nRisk Factors to Consider: \n- Regulatory changes (e.g., antitrust lawsuits) → Higher risk (4-5). \n- Product launches (e.g., VisionPro release) → Moderate risk (3). \n- Market sentiment (e.g., price might decrease) → Score 2-4 depending on certainty. \n- Ambiguous statements (e.g., might get the Chinese affect) → Default to 3 with low probability. \nRisk 1: Very low risk - Strong positive news (e.g., 45% stock increase with no negatives). \nRisk 3: Moderate risk - Vague or neutral news (e.g., 'might decrease'). \nRisk 5: Very high risk - Clear negative news (e.g., major scandal or bankruptcy risk). \nProbability Guidance: \n- Probability = 0.9-1.0 if the news cites a definitive event (e.g., FDA approval delayed). \n- Probability = 0.6-0.8 if the risk is implied but not certain (e.g., may face fines). \n- Probability < 0.5 if the news is vague (e.g., could be impacted by market trends). \nResponse Format: Risk: [risk score] \nProbability: [probability]"},
        {"role": "user",
         "content": f"News to Stock Symbol -- AMZN: Amazon warehouse strike begins ### News to Stock Symbol -- GOOGL: Google fined $1B for data privacy ### News to Stock Symbol -- JPM: JP Morgan beats earnings forecasts"},
         {"role": "assistant", "content": f"Risk: 3 \nProbability: 0.55, Risk: 5, \nProbability: 0.80, Risk: 1 \nProbability: 0.95"}, 
            {"role": "user",
            "content": "News to Stock Symbol -- AAPL: Apple faces antitrust lawsuit in the EU ### News to Stock Symbol -- TSLA: Tesla's Q4 deliveries miss estimates"},
            {"role": "assistant", "content": f"Risk: 4 /nProbability: 0.75, Risk: 4 \nProbability: 0.60"}, {"role": "user", "content": f"### News to Stock Symbol -- {row['Stock_symbol']}: On {row['Date']}, {row['Lsa_summary']}"}
    ]
        })

print(f"Number of prepared requests: {len(prepared_requests)}")

# Load the configuration
with open("sentiment_analysis.yml", "r") as f:
    config_dict = yaml.safe_load(f)
config = ParallelAPIRequesterConfig(**config_dict)

# Initialize the ParallelDeepSeekRequester
#llm_sentiment_gpt35 = ParallelDeepSeekRequester(name="deepseek-chat", config=config)

# Submit all the requests and get the responses
results = get_responses_parallel(prepared_requests, config=config)

# Process the results and merge with the original data
output_data = []
for request, result in zip(prepared_requests, results):
    # Extract risk and probability from the result
    risks, probabilities = extract_answer(result.get("response", ""))
    risk = risks[0] if risks else None
    probability = probabilities[0] if probabilities else None

    # Append the original data with the new columns
    output_data.append({
        "index": request["index"],
        "Stock_symbol": request["symbol"],
        "Date": request["date"],
        "Lsa_summary": request["lsa_summary"],
        "Risk": risk,
        "Probability": probability
    })

# Convert the output data to a DataFrame
output_df = pd.DataFrame(output_data)

# Merge with the original input data to preserve all columns
input_df = pd.read_csv(INPUT_FILE, encoding="utf-8", on_bad_lines='warn', engine='python')
final_df = input_df.merge(output_df, on=["Stock_symbol", "Date", "Lsa_summary"], how="left")

# Write the results to the output CSV
header = not os.path.exists(OUTPUT_FILE)
final_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)

logging.info("Script completed.")
print("Results written to:", OUTPUT_FILE)