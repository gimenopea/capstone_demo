import pandas as pd
import numpy as np
import openai
import os
import random
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from faker import Faker
import logging
import json
from langchain import PromptTemplate
from datetime import datetime
import sqlite3
import re

Faker.seed(2023)
fake = Faker()

def setup():
    load_dotenv()
    print('keys loaded')


    
setup()

llm = OpenAI(temperature=0.5)


allocations = {"Student Scholarships Fund": "Financial aid for outstanding or needy students.",
"Faculty Research Grant" : "Funding for faculty research projects.",
"Campus Sustainability Initiative:": "Support for environmental sustainability on campus.",
"Diversity and Inclusion Program": "Programs for promoting diversity and inclusion.",
"Infrastructure Upgrade Fund": "Funding for campus facilities and infrastructure maintenance and upgrades."}

dataset_characteristics = {"balanced features": "The dataset has a balanced distribution of features.",
"slightly skewed features on donor age": "The dataset has a slightly skewed distribution of features on donor age.",
"slightly skewed features on distribution of gifts": "The dataset has a slightly skewed distribution of features on distribution of gifts.",
"slightly skewed features on demographic representation": "The dataset has a slightly skewed distribution of features on demographic representation.",
"cold start training data samples": "The dataset may not have enough training data samples to train a model."}

donor_types = ["annual giving", "major gift"]

def random_dataset(dataset_characteristics):
    key, value = random.choice(list(dataset_characteristics.items()))
    

    #random string that has characters 2-3 characters and 5 digits and starts with the word 'dataset-'
    dataset_name = 'dataset-' + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 2)) + ''.join(np.random.choice(list('0123456789'), 5))

    #random dataset date that is between today and 7 days ago
    dataset_date = pd.Timestamp.today() - pd.Timedelta(days=np.random.randint(0, 7))
    dataset_date = dataset_date.strftime('%Y-%m-%d')
    cleared_for_access = np.random.choice([True, False])
    result = {
        'dataset_name': dataset_name,
        'dataset_date': dataset_date,
        'cleared_for_access': cleared_for_access,
        'dataset_profile': value
    }
    return result

def random_classification_model(donor_types):
    #get a random value for donor_types
    donor_type = np.random.choice(donor_types).item()
    model_name = 'model-' + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 2)) + ''.join(np.random.choice(list('0123456789'), 5))
    model_name  = model_name + '-' +np.random.choice(['stacking','boosting', 'pytorch'],1).item()
    model_date = pd.Timestamp.today() - pd.Timedelta(days=np.random.randint(0, 7))
    model_date = model_date.strftime('%Y-%m-%d')
    #create variable called accuracy values between .5 to .99

    accuracy = round(0.8 + 0.2 * np.random.rand(), 2)
    f1 = round(np.random.rand(),2)

    result = {
        'model_name': model_name,
        'model_date': model_date,
        'donor_type_classification': donor_type,
        'model_accuracy': accuracy,
    }
    return result
    
def donor_activity():
    #get first key, value pair from allocations dictionary
    key, value = random.choice(list(allocations.items()))
    allocation_name = key
    allocation_description = value
    amount = np.random.randint(100, 10000)
    donor_id = ''.join(np.random.choice(list('012'), 3))
    #create a fake donor name
    donor_name = fake.name()
    result = {'donor_id': donor_id,
              'allocation_name': allocation_name,
              'allocation_description': allocation_description,
              'amount': amount,
              'activity_type': 'donation'
    }

    
    return result

def evaluate_action_item(action_item):
    decision_grid = {
        ("High", "High"): "Do it now",
        ("High", "Low"): "Delegate it",
        ("Low", "High"): "Do it quickly",
        ("Low", "Low"): "Drop it"
    }
    
    decision = decision_grid.get((action_item["impact"], action_item["urgency"]), "Do it later")
    return {'Task': action_item['description'], 'Action': f'{decision}'}

def choose_task():
    tasks = ['send a thank you email', 'send a thank you letter', 'send a thank you text message', 'email your boss']
    return random.choice(tasks)
    

def generate_action_item():
    
    action_item = {
        "description": choose_task(),
        "impact": np.random.choice(["High", "Low"]),
        "urgency": np.random.choice(["High", "Low"])
    }
    action = evaluate_action_item(action_item)
    
    return action

def generate_payload():
    payload = {
    'activity': donor_activity(),
    'classification': random_classification_model(donor_types),
    'dataset': random_dataset(dataset_characteristics),
    'action': generate_action_item()}
    return payload

my_responsibility_profile = {'allocation_responsible': "i'm responsible for student funds, scholarships, and financial aid funds.",
                             'donor_type_responsible': "i'm responsible for major gift donors.",
                             'model_confidence_threshold': "i'm confident in models with an accuracy of 0.8 or higher.",
                             'dataset_threshold': "when dataset is cleared for access, i will follow through. If dataset is not cleared for access, i will e-mail staff for clearance."
                             }



def action_recommendation(payload):
    
    trail = dict()
    trail['request_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    llm_dict = {
    "llm_model_name": llm.model_name,
    "llm_temperature":llm.temperature,
    "llm_max_tokens" : llm.max_tokens,
    "llm_top_p": llm.top_p,
    "llm_frequency_penalty": llm.frequency_penalty,
    "llm_best_of": llm.best_of,
    "llm_batch_size": llm.batch_size }

    trail['llm_metadata'] = llm_dict
    
    template_action = """I want you to act as an advisor. Write a brief action phrase given the following information:
    task: {task},
    action: {action} 
    and the donor name is {donor_id}.
    """
    

    prompt = PromptTemplate(input_variables=["task", "action", "donor_id"], template=template_action)
    
    action = payload.get('action').get('Action')
    task = payload.get('action').get('Task')
    donor_id = payload.get('activity').get('donor_id')
    
    trail['prompt_chain_1'] = prompt
    
    p1 = prompt.format(task=task, action=action, donor_id=donor_id)
    recommendation = llm(p1)
    
    trail['prompt_chain_1_response'] = p1
    
    #block for explainer to the user
    
    model_name = payload.get('classification').get('model_name')
    model_accuracy = payload.get('classification').get('model_accuracy')
    model_date = payload.get('classification').get('model_date')
    
    if task == 'email your boss':
        detail = f'Dear Boss, check our donor {donor_id} and see if we can get a meeting with them.'
    else:
        prompt = f"""I want you to {task} to donor {donor_id}. given the urgency of: {action}"""
        detail = llm(prompt)
        trail['prompt_chain_2'] = prompt
        trail['prompt_chain_2_response'] = detail
        
    trail['model_explainer'] = f"The model {model_name} was trained on a dataset from {model_date} with an accuracy of {model_accuracy}. {recommendation} was determined the best course of action based on your goals"
    
    trail['detail'] = detail
    trail['payload'] = payload
    
    trail['my_responsibility_profile'] = my_responsibility_profile
    
    return recommendation, detail, trail


if __name__ == "__main__":
    conn = sqlite3.connect('trail.db')
    
    recommendation, explainer, trail = action_recommendation(generate_payload())    
    df = pd.DataFrame.from_dict([trail])    
    
    n_llm = pd.json_normalize(df['llm_metadata'])
    n_payload = pd.json_normalize(df['payload'])
    n_responsibility = pd.json_normalize(df['my_responsibility_profile'])

    df = pd.concat([df, n_llm, n_payload, n_responsibility], axis=1)

    #drop the columns that are nested dictionaries

    df.drop(['llm_metadata', 'payload','my_responsibility_profile'], axis=1, inplace=True)
    df['recommendation'] = recommendation 
    df['event_desc'] = f'event: {df["activity.activity_type"].loc[0]} made by donor id {df["activity.donor_id"].loc[0]}'
    df['id'] = df["request_time"].loc[0] + df["activity.donor_id"].loc[0] 
    df['id'] = re.sub(r'\W+', '', df['id'].loc[0])
    df['id'] = df['id'].apply(lambda x: x[1:] if x.startswith('0') else x)
    
    
    df = df.astype(str)

    df.to_sql('trail', conn, if_exists='append', index=True)      
    
    conn.close()
    
