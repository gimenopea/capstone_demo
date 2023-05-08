import random
from datetime import datetime
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


def setup():
    load_dotenv()
    print('keys loaded')


setup()
llm = OpenAI(temperature=0.9)


class DonorEngagement:
    def __init__(self, donor_data):
        self.donor_data = donor_data
        self.recommendations = self.generate_recommendations()
        self.event_type = donor_data['event_type']

    def generate_recommendations(self):
        recommendations_list = [
            "Personalized communication",
            "Thank-you call or letter",
            "Share impact stories",
            "Invite to events",
            "Offer volunteering opportunities",
            "Recognition",
            "Establish a personal connection",
            "Solicit feedback",
            "Encourage peer-to-peer fundraising",
            "Stewardship"
        ]

        num_recommendations = 2
        chosen_recommendations = random.sample(
            recommendations_list, num_recommendations)

        bullet_point_recommendations = [
            f"{i+1}. {rec}" for i, rec in enumerate(chosen_recommendations)]

        return bullet_point_recommendations

    def generate_llm(self):
        donor_payload = self.donor_data
        task = self.recommendations
        trail = dict()
        trail['request_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        llm_dict = {
            "llm_model_name": llm.model_name,
            "llm_temperature": llm.temperature,
            "llm_max_tokens": llm.max_tokens,
            "llm_top_p": llm.top_p,
            "llm_frequency_penalty": llm.frequency_penalty,
            "llm_best_of": llm.best_of,
            "llm_batch_size": llm.batch_size}

        trail['llm_metadata'] = llm_dict

        if self.event_type == 'donation':
            template_action = """given the following tasks and additional context, write a personalized communication to donor to accomplish it:
            input task: {task},
            input context: {donor_payload}
            """
            prompt = PromptTemplate(
                input_variables=['task', 'donor_payload'], template=template_action)
            p1 = prompt.format(task=task, donor_payload=donor_payload)

        elif self.event_type == 'prospect_interaction':
            urgency = donor_payload.get('urgency')
            # Changed 'classification' to 'donor_classification'
            donor_classification = donor_payload.get('donor_classification')
            interaction = donor_payload.get('interaction')

            template_action = """given the following interaction and additional context, write a personalized communication to donor to accomplish it:
            input interaction: {interaction},
            input urgency: {urgency},
            input classification: {donor_classification}
            """

            prompt = PromptTemplate(input_variables=[
                                    'interaction', 'urgency', 'donor_classification'], template=template_action)
            p1 = prompt.format(interaction=interaction, urgency=urgency,
                               donor_classification=donor_classification)

        trail['prompt_chain_1'] = template_action
        recommendation = llm(p1)

        trail['prompt_chain_1_response'] = p1

        return recommendation, trail


class EmployeeProfile:
    def __init__(self, donor_data):
        self.donor_data = donor_data

    def employee_interests(self):
        payload = {'classification': self.donor_data['action'].get('donor_classification'),
                   'urgency': self.donor_data['action'].get('urgency'),
                   'recommendation': self.donor_data.get('recommendation'),
                   'donation_purpose': self.donor_data['action'].get('donation_purpose'),
                   'donation_category': self.donor_data['action'].get('gift_category'),
                   'donation_description': self.donor_data['action'].get('description')}
        return payload

    def generate_llm_goals(self):
        donor_payload = self.employee_interests()
        template_action = """based on this input context of a prospect, 4-5 bullet point list of what a fundraiser goal might be:
            input context: {donor_payload}
        """
        prompt = PromptTemplate(
            input_variables=['donor_payload'], template=template_action)
        p1 = prompt.format(donor_payload=donor_payload)

        return llm(p1)

    def generate_llm_dataset(self):
        data_payload = self.donor_data['dataset']
        data_payload['dataset_characteristics'] = self.donor_data.get(
            'dataset_characteristics')

        template_action = """given the following input, summarize the dataset and write a 4-5 bullet point goal for a data steward managing this process:
            input context: {data_payload}
        """
        prompt = PromptTemplate(
            input_variables=['data_payload'], template=template_action)
        p1 = prompt.format(data_payload=data_payload)

        return llm(p1)
