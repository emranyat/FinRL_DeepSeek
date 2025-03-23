# Deepinfra file

import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
import re

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
PATTERN_SEP = r"\n(.*\n)*?"
PATTERN_ANSWER = r".+"
PATTERN_FLOAT = r"\d*\.?\d+"
PROMPT_EXAMPLE_FIVE_SHOT_SCORES = [0.80, 0.43, 0.71, 0.34, 0.08] # redo it as needed
response_patterns = [
        re.compile(fr"(Risk: |Answer: )(?P<score>{PATTERN_ANSWER}){PATTERN_SEP}Probability: (?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Risk: |Answer: )?(?P<score>{PATTERN_ANSWER})\n+(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Risk: |Answer: )?(?P<score>{PATTERN_ANSWER})(, |. |; | - | \| )(Probability: )?(?P<confidence>{PATTERN_FLOAT})"),
        re.compile(fr"(Risk: |Answer: )?(?P<score>{PATTERN_ANSWER}) \((Probability: )?(?P<confidence>{PATTERN_FLOAT})\)"),
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
    if isinstance(names, str):
        return None
    else:
        return (None for _ in names)
def extract_answer(responses):
    risks, confprobs = [], []
    for response in responses.split(','): 
        score, confidence = extract_from_response(response, response_patterns, ("score", "confidence"))
        try:
                risk_value = int(score.strip())
                conf_value = int(confidence.strip())
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
    return risks, confprobs
    
def get_risk(symbol, date, *texts):
    texts = [text for text in texts if text != 0]
    num_text = len(texts)
    text_content = " ".join([f"### News to Stock Symbol -- {symbol}: {text}" for text in texts])
    #print(text_content)
    #Provide your best guess and the probability between 0.0 and 1.0 that your best guess is correct or plausible for the given task. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Use the following format to respond:\n```\nGuess: [most likely guess, as short as possible, only the guess]\nProbability: [probability between 0.0 and 1.0 that your guess is correct, without comments, only the probability]\n```\n{PROMPT_NO_ANSWER} {self.PROMPT_EXAMPLE_FIVE_SHOT}",
    conversation = [
        {"role": "system",
         "content": f"Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations in the date of {date}. You have no knowledge of events, prices, or news after the date of {date}. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. {num_text} summarized news will be passed in each time. Also provide your probability that your risk score is correct between 0.00 and 1.00. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Use the following format to respond:\n```\nRisk: [most likely risk score, as a single number]\nProbability: [probability between 0.0 and 1.0 that your risk score is correct, without comments, only the probability]. Provide the score and the probability in the format shown below in the response from the assistant."},
        {"role": "user",
         "content": f"News to Stock Symbol -- AAPL: Apple (AAPL) increases 45% ### News to Stock Symbol -- AAPL: Apple (AAPL) price might decrease ### News to Stock Symbol -- MSFT: Microsoft (MSFT) price has no change"},
        {"role": "assistant", "content": f"Risk: 1 \nProbability: 0.80, Risk: 5 \nProbability: 0.43, Risk: 3 \nProbability: 0.71"},  # Risk assessment applied: no major risk indication for 22% increase, high risk for 30% decrease, neutral for no change.
        {"role": "user",
         "content": f"News to Stock Symbol -- TSLA: Tesla (TSLA) might get the Chinese affect  ### News to Stock Symbol -- AAPL: Apple (AAPL) will release VisionPro on Feb 2, 2024"},
        {"role": "assistant", "content": "Risk: 4 \nProbability: 0.34, Risk: 3 \nProbability: 0.08"},  # Risk assessment: no significant indication of risk in the announcements, so both scores are 3.
        {"role": "user", "content": text_content},
    ]
    

    #risks = []
    #confprobs = []
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
    
    
    risks, confprobs = extract_answer(content)
    
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
    return risks, confprobs

def from_csv_get_risk(df, saving_path, batch_size=4):
    df.sort_values(by=model_used, ascending=False, na_position='last', inplace=True)
    if 'New_text' in df.columns:
        df.rename(columns={'New_text': 'Lsa_summary'}, inplace=True)
    if 'Date' not in df.columns: # Check if the 'Date' column is present in the DataFrame
        raise ValueError("The 'Date' column is missing from the DataFrame.")
    for i in range(0, len(df), batch_size):
        if df.loc[i:min(i + batch_size - 1, len(df) - 1), model_used].notna().all():
            continue
        print("Now row: ", i)
        texts = [df.loc[j, 'Lsa_summary'] if j < len(df) else 0 for j in range(i, i + batch_size)]
        symbol = df.loc[i, 'Stock_symbol']  # Extract the stock symbol for the current batch
        date = df.loc[i, 'Date'] # I added this to make sure it does not use info after this date
        if pd.isnull(date):
            print(f"Skipping row {i} due to missing or invalid date.")
            continue
        print(f"Processing row {i} with date: {date}")

        risks, confprobs = get_risk(symbol, *texts, date)

        for k, risk in enumerate(risks):
            if i + k < len(df):
                df.loc[i + k, model_used] = risk
        for k, conf in enumerate(confprobs):
            if i + k < len(df):
                df.loc[i + k, model_used+'conf'] = conf
                
        df.to_csv(saving_path, index=False)  # Save the entire DataFrame with all columns
    return df


def process_csv(input_csv_path, output_csv_path, batch_size=5, chunk_size=1000):
    start_time = time.time()

    # Check if the output file exists and load the last processed row
    if os.path.exists(output_csv_path):
        output_df = pd.read_csv(output_csv_path, 
        on_bad_lines='warn',
        engine='python'
)
        last_processed_row = len(output_df)
    else:
        last_processed_row = 0

    # Read the CSV file in chunks
    chunks = pd.read_csv(input_csv_path, encoding="utf-8", chunksize=chunk_size,
    on_bad_lines='warn', 
    engine='python'     # Print a warning for each skipped line
    )

    for chunk_number, chunk in enumerate(chunks):
        # Skip already processed chunks
        if chunk_number * chunk_size < last_processed_row:
            continue

        chunk.columns = chunk.columns.str.capitalize()
        if model_used not in chunk.columns:
            chunk[model_used] = np.nan

        for i in range(0, len(chunk), batch_size):
            global batch
            batch = chunk.iloc[i:i + batch_size] 
            #print('the symbol == ', batch.columns) 
            texts = batch['Lsa_summary'].tolist()
            symbol = batch.iloc[0]['Stock_symbol']  # Extract the stock symbol for the current batch
            date = batch.iloc[0]['Date']
            #print(date)
            risks, confprobs = get_risk(symbol, date, *texts)

            for j, risk in enumerate(risks):
                if i + j < len(chunk):
                    chunk.loc[chunk.index[i + j], model_used] = risk
            for j, conf in enumerate(confprobs):
                if i + j < len(chunk):
                    chunk.loc[chunk.index[i + j], model_used+'conf'] = conf
                    
        # Append the processed chunk to the output file
        chunk.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

    print(f"Process completed in {time.time() - start_time:.2f} seconds.")
    


if __name__ == "__main__":
    input_file = 'news_llm_sent_AAPL.csv'
    output_file = model_used + '_' + input_file
    process_csv(
        rf'C:\Users\lenovo-pc\Documents\GitHub\FinRL_DeepSeek\{input_file}', 
        rf'C:\Users\lenovo-pc\Documents\GitHub\FinRL_DeepSeek\{output_file}', 
        batch_size=4
    )
