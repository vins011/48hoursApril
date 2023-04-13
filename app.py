from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import openai
import re
import fitz
import tiktoken as tk
import json
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime

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
    ts = datetime.now ()
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
@app.route('/assist', methods=['POST'])
def getSummery():
    data = request.get_json()
    #print(data)
    input_text = data['input']
    if (('hi' == input_text.lower()) or ('hello' == input_text.lower())):
        return jsonify({'result':'How can I assist you today, I am here to generate summary for your given input..'})
    openai.api_key = "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"
    prompt = prompt_lookup(input_text)
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
    return jsonify({'result': summary})

# Example for transcript
@app.route('/uploadAndTrainData', methods=['POST'])
def getScript():
    model_id = 'whisper-1'
    date = request.form.get('date');
    agenda = request.form.get('agenda');
    organiser = request.form.get('Organiser');
    ts = datetime.now()
    
    print("Posted file: {}".format(request.files['files']))
    audio_file = request.files['files']
    currentWorkingDirectory = [os.getcwd(), "\\uploads\\", ts.strftime("%m%d%Y_%H%M%S"), "_" , audio_file.filename ]
    file_path = "".join(currentWorkingDirectory)
    audio_file.save(file_path)

    
    file_ext = os.path.splitext(audio_file.filename)[1]

    if file_ext == ".pdf":
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            summary = text
            print(summary)
        response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        api_key = "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",
        messages=[{'role':'system','content':'You are good at creating bullet point summaries and have knowledge of AWS '},
        {'role':'user','content':f"Summarize the following in bullet point form:\n{summary}"}
        ]
        )
    else:
        audio_file1 = open(file_path, "rb")
        response = openai.Audio.transcribe(
        api_key = "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",
        model = model_id,
        file= audio_file1)
        summary = response
        print(summary)
        response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        api_key = "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",
        messages=[{'role':'system','content':'You are good at creating bullet point summaries and have knowledge of AWS '},
        {'role':'user','content':f"Summarize the following in bullet point form:\n{summary['text']}"}
        ]
        )
    #print(summary)
    train_data(f"Agenda of this meeting is {agenda}. Organised or authour is/are:  {organiser}. this meeting held on: {date} content: {response.choices[0].message.content}")
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
    #train_data(f"purpose: aws discussion organisation: sample unicorn, content:{response.choices[0].message.content}, date:20-12-2022")
    return response.choices[0].message.content
    

if __name__ == '__main__':
    app.run(debug=True)
