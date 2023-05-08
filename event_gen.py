from activity import DonationGenerator, FictionalDatasetMetadata, ModelTrainingMetadata, ProspectInteractionGenerator
from skynet import DonorEngagement, EmployeeProfile
from sqlalchemy import create_engine
from datetime import datetime
import sqlite3
import pandas as pd
import random
import re

from dotenv import load_dotenv


def setup():
    load_dotenv()
    print('keys loaded')


setup()


purposes = {"Student Scholarships Fund": "Financial aid for outstanding or needy students.",
            "Faculty Research Grant": "Funding for faculty research projects.",
            "Campus Sustainability Initiative:": "Support for environmental sustainability on campus.",
            "Diversity and Inclusion Program": "Programs for promoting diversity and inclusion.",
            "Infrastructure Upgrade Fund": "Funding for campus facilities and infrastructure maintenance and upgrades."}


def run():

    conn = sqlite3.connect('trail.db')
    trail = dict()

    # create 1 donation event
    donation_generator = DonationGenerator(purposes)
    donation_event = [donation_generator.generate_donation_event()
                      for _ in range(1)]
    donation_event = donation_event[0]

    prospect_interaction = ProspectInteractionGenerator().generate_interaction()

    trail['action'] = random.choice([donation_event, prospect_interaction])

    donor_engagement = DonorEngagement(trail['action'])

    if trail['action'].get('event_type') == 'prospect_interaction':
        trail['recommendation'] = f"{trail['action'].get('interaction')}"
    else:
        trail['recommendation'] = f"The model wants you to prioritize on: {' '.join(donor_engagement.recommendations)}"

    trail['recommendation_detail'], trail['llm_chain'] = donor_engagement.generate_llm()

    dataset_used = FictionalDatasetMetadata()

    trail['dataset'] = dataset_used.metadata
    trail['model'] = ModelTrainingMetadata().metadata
    trail['dataset']
    dataset_characteristics = ["balanced features: The dataset has a balanced distribution of features.",
                               "slightly skewed features on donor age: The dataset has a slightly skewed distribution of features on donor age.",
                               "slightly skewed features on distribution of gifts The dataset has a slightly skewed distribution of features on distribution of gifts.",
                               "slightly skewed features on demographic representationThe dataset has a slightly skewed distribution of features on demographic representation.",
                               "cold start training data samples The dataset may not have enough training data samples to train a model."]
    trail['dataset_characteristics'] = random.choice(dataset_characteristics)

    employee_profile = EmployeeProfile(trail)

    trail['employee_goals'] = employee_profile.generate_llm_goals()
    trail['employee_datasteward'] = employee_profile.generate_llm_dataset()
    trail['request_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    trail['eventid'] = str(trail['action'].get(
        'prospect_id')) + trail['request_time']

    trail['eventid'] = ''.join(filter(str.isalnum, trail['eventid']))
    df = pd.json_normalize(trail)

    df.to_sql('trail', con=conn, if_exists='append', index=False)

    return trail, df
