# Deepinfra file

import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
MAX_WORKERS = 5  # Adjust based on your API rate limits
from tenacity import retry, stop_after_attempt, wait_exponential

# Set your OpenAI API key and base URL
openai = OpenAI(
    api_key= "sk-7c563985e05342108b5f27245c59b167", # Replace with your actual DeepInfra token
    base_url="https://api.deepseek.com",
)
stream = False  # Set to True if you want to stream the response

# Define the model_used variable
#model_used = 'risk_Llama-3.3-70B-Instruct'

model_used='risk_deepseek'

# I need to fix those , espicially this PATTERN_ANSWER
NO_ANSWER_TEXT = "NO ANSWER"
PROMPT_NO_ANSWER = f"If you cannot provide an answer, answer with `{NO_ANSWER_TEXT}`."
PATTERN_SEP = rf"\n(.*\n)*?"
PATTERN_ANSWER = rf"\b[1-5]\b" # I need to fix this to be a number INT
PATTERN_FLOAT = rf"\d*\.?\d+"
PROMPT_EXAMPLE_FIVE_SHOT_SCORES = [0.80, 0.43, 0.71, 0.34, 0.08] # redo it as needed

PATTERN_ANSWER = r"\b[1-5]\b"  # Ensures risk scores are integers 1-5
response_patterns = [
    re.compile(fr"Risk:?\s*(?P<score>{PATTERN_ANSWER})\D+(Probability:?\s*)?(?P<confidence>{PATTERN_FLOAT})", re.IGNORECASE),
    re.compile(fr"Answer:?\s*(?P<score>{PATTERN_ANSWER}).*?Probability:?\s*(?P<confidence>{PATTERN_FLOAT})", re.IGNORECASE),
]
# I added them from the prompt conf paper
def normalize_confidence(confidence, normalize_fn):
    if confidence is not None:
        confidence = normalize_fn(confidence)
    return confidence

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
    

""" def extract_answer(responses):
    risks, confprobs = [], []
    for response in responses.split(','): 
        score, confidence = extract_from_response(response, response_patterns, ("score", "confidence"))
        try:
            risk_value = int(score.strip()) if score else np.nan
            conf_value = float(confidence.strip()) if confidence else np.nan
        except ValueError:
                print("content error")
                print(' content is: ' + str(score.strip()))
                risk_value = np.nan
                print(' content is: ' + str(confidence.strip()))
                conf_value = np.nan
            #confidence = normalize_confidence(confidence)           
            # filter out confidence scores used in few-shot examples
        if confidence in PROMPT_EXAMPLE_FIVE_SHOT_SCORES:
                confidence = None
        risks.append(score)
        confprobs.append(confidence)
    #print('this risks from extract answer function == ', risks)
    return risks, confprobs """

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
# add retry 
#@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))    
def get_risk(symbol, date, text):
    #texts = [text for text in texts if text != 0]
    #num_text = len(texts) #{num_text} summarized news will be passed in each time.
    text_content = f"### News to Stock Symbol -- {symbol}: {text}"
    ##text_content = " ".join([f"### News to Stock Symbol -- {symbol}: {text}" for text in texts])
    conversation = [
        {"role": "system",
         "content": f"Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations in the date of {date}. You have no knowledge of events, prices, or news after the date of {date}. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. Also provide your probability that your risk score is correct between 0.00 and 1.00. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Use the following format to respond:\n```\nRisk: [most likely risk score, as a single number]\nProbability: [probability between 0.0 and 1.0 that your risk score is correct, without comments, only the probability]. Provide the score and the probability in the format shown below in the response from the assistant."},
        {"role": "user",
         "content": f"News to Stock Symbol -- AAPL: Apple (AAPL) increases 45% ### News to Stock Symbol -- AAPL: Apple (AAPL) price might decrease ### News to Stock Symbol -- MSFT: Microsoft (MSFT) price has no change"},
        {"role": "assistant", "content": f"Risk: 1 \nProbability: 0.80, Risk: 5 \nProbability: 0.43, Risk: 3 \nProbability: 0.71"},  # Risk assessment applied: no major risk indication for 22% increase, high risk for 30% decrease, neutral for no change.
        {"role": "user",
         "content": f"News to Stock Symbol -- TSLA: Tesla (TSLA) might get the Chinese affect  ### News to Stock Symbol -- AAPL: Apple (AAPL) will release VisionPro on Feb 2, 2024"},
        {"role": "assistant", "content": "Risk: 4 \nProbability: 0.34, Risk: 3 \nProbability: 0.08"},  # Risk assessment: no significant indication of risk in the announcements, so both scores are 3.
        {"role": "user", "content": text_content},
    ]
    

    risks = []
    confprobs = []
    try:
        chat_completion = openai.chat.completions.create(
          #  model="meta-llama/Llama-3.3-70B-Instruct",
      #      model="Qwen/Qwen2.5-72B-Instruct",
            model='deepseek-chat',
            messages=conversation,
            temperature=0,
            max_tokens=50,
            stream=stream,
        )

        if stream:
            content = ""
            for event in chat_completion:
                if event.choices[0].finish_reason:
                    print(event.choices[0].finish_reason,
                          event.usage['prompt_tokens'],
                          event.usage['completion_tokens'])
                else:
                    content += event.choices[0].delta.content
            #print(content)
        else:
            content = chat_completion.choices[0].message.content
            print(content)
            print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
        #return risks[0] if risks else np.nan, confprobs[0] if confprobs else np.nan
    
    except AttributeError:
        print("response error")
        risk_value = np.nan
        risks.append(risk_value)
        confprobs.append(risk_value)
        return risks, confprobs 
    except Exception as e:
        print(f"Error: {e}")
        risk_value = np.nan
        risks.append(risk_value)
        confprobs.append(risk_value)
        return risks, confprobs
    # After extracting content:
    print(f"Raw API Response: {content}")  # Debug output
    risks, confprobs = extract_answer(content)
    print(f"Parsed Risks: {risks}, Confidences: {confprobs}")  # Debug parsed values
    #risks, confprobs = extract_answer(content)
    
#     for risk in content.split(','):
#         try:
#             risk_value = int(risk.strip())
#         except ValueError:
#             print("content error")
#             print(' content is: ' + str(risk.strip()))
#             risk_value = np.nan
#     risks.append(risk)
    
#     for conf in confidence.split(','):
#         try:
#             conf_value = int(conf.strip())
#         except ValueError:
#             print("content error")
#             print(' content is: ' + str(conf.strip()))
#             conf_value = np.nan
#     confprobs.append(confidence)
#     print(risks, cpnfprobs)
    #print('this risks from get_risk function == ', risks)
    return risks, confprobs


def process_csv(input_csv_path, output_csv_path, batch_size=5, chunk_size=1000):
    start_time = time.time()

    # Check if the output file exists and load the last processed row
    if os.path.exists(output_csv_path):
        output_df = pd.read_csv(output_csv_path, 
        on_bad_lines='warn',
        engine='python'
)
#        processed_indices = set(output_df.index)
#    else:
#        processed_indices = set()
        
        last_processed_row = len(output_df)
    else:
        last_processed_row = 0

    # Read the CSV file in chunks
    chunks = pd.read_csv(input_csv_path, encoding="utf-8", chunksize=chunk_size,
    on_bad_lines='warn', 
    engine='python'     # Print a warning for each skipped line
    )
    # new from here
# here the new thing i took out ######################
    for chunk in chunks:
        chunk = chunk.reset_index(drop=True)
        if model_used not in chunk.columns:
            chunk[model_used] = np.nan
        if model_used+'conf' not in chunk.columns:
            chunk[model_used+'conf'] = np.nan
        for i in range(len(chunk)):
            if pd.notna(chunk.loc[i, model_used]):
                continue
            text = chunk.loc[i, 'Lsa_summary']
            symbol = chunk.loc[i, 'Stock_symbol']
            date = chunk.loc[i, 'Date']
            risks, confprobs = get_risk(symbol, date, text)
            chunk.loc[i, model_used] = risks[0] if risks else np.nan
            chunk.loc[i, model_used+'conf'] = confprobs[0] if confprobs else np.nan

  
            
            # Update output file existence status
        if not os.path.exists(output_csv_path):
            open(output_csv_path, 'w').close()   
            # Save progress after each batch
            #chunk.to_csv(output_csv_path, index=False)
            ##print(f"Processed batch {i//batch_size}: Risks={risks}")

        # Append the processed chunk to the output file
        header = not os.path.exists(output_csv_path)  # Write header only once
        chunk.to_csv(output_csv_path, mode='a', header=header, index=False)

    print(f"Process completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    input_file = 'news_llm_sent_AAPL.csv'
    output_file = model_used + '2222_' + input_file
    process_csv(
        rf'C:\Users\lenovo-pc\Documents\GitHub\FinRL_DeepSeek\{input_file}', 
        rf'C:\Users\lenovo-pc\Documents\GitHub\FinRL_DeepSeek\{output_file}', 
        batch_size=4
    )
