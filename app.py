from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import openai
import re
import tiktoken as tk
import json
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS (app) 

# Example endpoint for GET requests
@app.route('/hello')
def hello():
    name = request.args.get('name', default='World')
    message = f'Hello, {name}!'
    return jsonify({'message': message})

# Example endpoint for POST requests
@app.route('/add', methods=['POST'])
def add():
    data = request.get_json()
    x = data['x']
    y = data['y']
    result = x + y
    return jsonify({'result': result})
    

def get_embedding(text):
    result = openai.Embedding.create (model='text-embedding-ada-002', input = text)
    return  result['data'][0]['embedding']

def num_tokens_from_string (string, encoding_name):
    encoding= tk.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def train_data (text):
    openai.api_key ="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"
    token = num_tokens_from_string (text, 'cl100k_base')
    embedding = get_embedding (text)
    data = {
        'summary': {text},
        'token': {token}
    }
    df = pd.json_normalize(data)
    output_path='my_csv.csv'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    df1 = pd.read_csv(output_path)
    df1['embedding']= df1['summary'].apply(get_embedding)

def prompt_lookup(text):
    openai.api_key ="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"
    output_path='my_csv.csv'
    df = pd.read_csv(output_path)
    df['embedding']= df['summary'].apply(get_embedding)
    question = text #input ("what question do you have about a unicorn startup ?")
    prompt_embedding = get_embedding (question)
    df ['prompt_similarity'] = df['embedding'].apply(lambda vector:vector_similarity(vector,prompt_embedding)) 
    context = df.nlargest (1, 'prompt_similarity').iloc[0]['summary']
    prompt = f'''context: {context}
    Q: {question} ? 
    A:'''
    
    return prompt

def vector_similarity (vec1, vec2):
    return np.dot (np.array(vec1),np.array(vec2))

# Example endpoint for POST requests
@app.route('/getSummary', methods=['POST'])
def getSummery():
    data = request.get_json()
    #print(data)
    input_text = data['input']
    prompt = prompt_lookup(input_text)
    if (('hi' == input_text.lower()) or ('hello' == input_text.lower())):
        return jsonify({'result':'How can I assist you today, I am here to generate summary for your given input..'})
    openai.api_key = "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"
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=1000,
    n=1,
    stop=None,
    temperature=0.5,
    )
    #print(response)
    summary = response.choices[0].text.strip()
    
    #summary = response
    #print(summary)
    #response = openai.ChatCompletion.create(
    #model='gpt-3.5-turbo',
    #api_key = "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",
    #messages=[{'role':'system','content':'You are good at creating bullet point summaries'},
    #{'role':'user','content':f"Summarize the following in bullet point form:\n{summary}"}
    #]
    #)
    #print(response)
    #return response.choices[0].message.content
    #summary = re.sub(r'\n', ' ', summary)
    #summary = re.sub(r'\s+', ' ', summary)
    #print(summary)
    #return jsonify({'result': response.choices[0].message.content})
    return jsonify({'result': summary})
    #return response.choices[0].message.content

# Example for transcript
@app.route('/getScript', methods=['GET'])
def getScript():
    model_id = 'whisper-1'
    file_path = "C:/Users/chirag/Downloads/AWS.mp3"
    #file_path = "C:/Users/chirag/Downloads/WAudio.mpeg"
    #You are good at creating bullet point summaries and have knowledge of upcoming vacation plan
    audio_file = open(file_path, "rb")
    response = openai.Audio.transcribe(
    api_key = "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",
    model = model_id,
    file= audio_file
    )
    summary = response
    #print(summary)
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    api_key = "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",
    messages=[{'role':'system','content':'You are good at creating bullet point summaries and have knowledge of AWS '},
    {'role':'user','content':f"Summarize the following in bullet point form:\n{summary['text']}"}
    ]
    )
    return response.choices[0].message.content

# Example for transcript
@app.route('/getScriptSummary', methods=['POST'])
def getScriptSummary():
    data = request.get_json()
    print(data)
    input_text = data['fileLocation']
    print(input_text)
    model_id = 'whisper-1'
    audio_file = open(input_text, "rb")
    response = openai.Audio.transcribe(
    api_key = "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",
    model = model_id,
    file= audio_file
    )
    summary = response
    #print(summary)
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    api_key = "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",
    messages=[{'role':'system','content':'You are good at creating bullet point summaries and have knowledge of AWS '},
    {'role':'user','content':f"Summarize the following in bullet point form:\n{summary['text']}"}
    ]
    )
    train_data(f"purpose: aws discussion organisation: sample unicorn, content:{response.choices[0].message.content}, date:20-12-2022")
    return response.choices[0].message.content
    

if __name__ == '__main__':
    app.run(debug=True)
