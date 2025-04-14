# Deepinfra file

import os
import time
import numpy as np
import pandas as pd
import re
from openai import OpenAI

# Set OpenAI API key and base URL
openai = OpenAI(
    api_key="sk-7c563985e05342108b5f27245c59b167",  # Replace with your actual DeepInfra token
    base_url="https://api.deepseek.com",
)
stream = False
model_used = 'risk_deepseek'

# Patterns for extracting risk and probability
PATTERN_ANSWER = r"\b[1-5]\b"
PATTERN_FLOAT = r"\d*\.?\d+"
response_patterns = [
    re.compile(fr"Risk:?\s*(?P<score>{PATTERN_ANSWER})\D+(Probability:?\s*)?(?P<confidence>{PATTERN_FLOAT})", re.IGNORECASE),
]

def extract_answer(response):
    """Extract risk and probability from the API response."""
    risks, confprobs = [], []
    # Split responses by "Risk:" delimiter
    entries = re.split(r'(?=\bRisk:\s*\d+)', response)
    for entry in entries:
        if not entry.strip():
            continue
        match = re.search(response_patterns[0], entry)
        risks.append(int(match.group("score")) if match and match.group("score") else np.nan)
        confprobs.append(float(match.group("confidence")) if match and match.group("confidence") else np.nan)
    return risks, confprobs

def get_risk(symbol, date, text):
    """Send a request to the API and extract risk and probability."""
    text_content = f"### News to Stock Symbol -- {symbol}: {text}"
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
    try:
        chat_completion = openai.chat.completions.create(
            model='deepseek-chat',
            messages=conversation,
            temperature=0,
            max_tokens=50,
            stream=stream,
        )
        # Handle streaming and non-streaming responses
        content = "".join(event.choices[0].delta.content for event in chat_completion) if stream else chat_completion.choices[0].message.content
        return extract_answer(content)
    except Exception as e:
        print(f"Error in get_risk for symbol {symbol} on {date}: {e}")
        return [np.nan], [np.nan]

def process_csv(input_csv_path, output_csv_path, chunk_size=1000):
    """Process the input CSV file and append risk and probability to the output."""
    start_time = time.time()

    # Check if the output file exists and load the last processed row
    if os.path.exists(output_csv_path):
        last_processed_row = len(pd.read_csv(output_csv_path, on_bad_lines='warn', engine='python'))
    else:
        last_processed_row = 0

    chunks = pd.read_csv(input_csv_path, encoding="utf-8", chunksize=chunk_size, on_bad_lines='warn', engine='python')
    for chunk in chunks:
        chunk = chunk.reset_index(drop=True)
        if model_used not in chunk.columns:
            chunk[model_used] = np.nan
            chunk[model_used + 'conf'] = np.nan

        for i, row in chunk.iterrows():
            if pd.notna(row[model_used]):
                continue
            # Use `or` to provide default values if any column is missing or NaN
            risks, confprobs = get_risk(
                row['Stock_symbol'] or "Unknown Symbol",
                row['Date'] or "Unknown Date",
                row['Lsa_summary'] or "No Summary"
            )
            chunk.at[i, model_used] = risks[0]
            chunk.at[i, model_used + 'conf'] = confprobs[0]
            print(risks)

        header = not os.path.exists(output_csv_path)
        chunk.to_csv(output_csv_path, mode='a', header=header, index=False)

    print(f"Process completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    input_file = 'news_llm_sent_AAPL.csv'
    output_file = model_used + '_processed_' + input_file
    process_csv(
        rf'C:\Users\lenovo-pc\Documents\GitHub\FinRL_DeepSeek\{input_file}', 
        rf'C:\Users\lenovo-pc\Documents\GitHub\FinRL_DeepSeek\{output_file}'
    )
