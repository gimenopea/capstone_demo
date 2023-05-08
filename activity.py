import random
import datetime


class DonationGenerator:
    def __init__(self, purposes):
        self.purposes = purposes

    def random_donor_id(self):
        return random.randint(1000, 9999)

    def random_purpose(self):
        return random.choice(list(self.purposes.keys()))

    def random_amount(self):
        # Adjust the range according to your requirements
        return random.uniform(1, 2000000)

    def categorize_gift(self, amount):
        if amount >= 1000000:
            return 'principal_gift'
        elif amount >= 50000:
            return 'major_gift'
        else:
            return 'annual_gift'

    def generate_donation_event(self):
        donor_id = self.random_donor_id()
        purpose = self.random_purpose()
        description = self.purposes[purpose]
        amount = self.random_amount()
        gift_category = self.categorize_gift(amount)
        return {
            'event_type': 'donation',
            'prospect_id': donor_id,
            'donation_purpose': purpose,
            'description': description,
            'amount': round(amount, 2),
            'gift_category': gift_category,
            'interaction_date': None,
            'interaction': None,
            'donor_classification': None,
            'urgency': None

        }


class FictionalDatasetMetadata:
    def __init__(self):
        self.metadata = self.generate_metadata()

    def generate_dataset_name(self):
        word = random.choice(['data', 'set', 'info', 'records', 'samples'])
        number = random.randint(1000, 9999)
        return f'{word}{number}'

    def random_date_between_now_and_last_year():
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365)
        delta = end - start
        random_days = random.randrange(delta.days)
        date = start + datetime.timedelta(days=random_days)
        return date.strftime('%Y-%m-%d')

    def generate_metadata(self):
       # creation_date = self.random_date_between_now_and_last_year()
        is_privileged = random.choice(['yes', 'no'])
        is_biased = random.choice(['biased', 'not_biased'])
        num_records = random.randint(50, 10000)
        avg_record_size = random.randint(100, 10000)
        num_columns = random.randint(1, 100)
        num_missing_values = random.randint(0, int(num_records * 0.2))
        data_format = random.choice(['CSV', 'JSON', 'Parquet', 'Avro'])
        data_source = random.choice(
            ['web_scraped', 'survey', 'API', 'internal'])
        data_storage = random.choice(['local', 'cloud'])
        data_domain = random.choice(
            ['donor_db', 'university_records', 'vendor_purchased', 'CRM', 'insider_source'])

        bias_reasons = [
            'Underrepresentation of certain groups',
            'Data collected from biased sources',
            'Biased sampling techniques',
            'Measurement errors',
            'Confounding variables not considered'
        ]
        reason_for_bias = random.choice(
            bias_reasons) if is_biased == 'biased' else None

        return {
            'dataset_name': self.generate_dataset_name(),
            # 'creation_date': creation_date,
            'is_privileged': is_privileged,
            'is_biased': is_biased,
            'num_records': num_records,
            'avg_record_size': avg_record_size,
            'num_columns': num_columns,
            'num_missing_values': num_missing_values,
            'data_format': data_format,
            'data_source': data_source,
            'data_storage': data_storage,
            'data_domain': data_domain,
            'reason_for_bias': reason_for_bias
        }


class ModelTrainingMetadata:
    def __init__(self):
        self.metadata = self.generate_metadata()

    def generate_dataset_name(self):
        word = random.choice(['data', 'set', 'info', 'records', 'samples'])
        number = random.randint(1000, 9999)
        return f'{word}{number}'

    def generate_metadata(self):
        learning_rate = random.uniform(0.0001, 0.1)
        batch_size = random.choice([16, 32, 64, 128, 256])
        num_epochs = random.randint(10, 200)
        optimizer = random.choice(['SGD', 'Adam', 'RMSprop', 'Adagrad'])
        loss_function = random.choice(
            ['CrossEntropyLoss', 'MSELoss', 'BCELoss', 'MAELoss'])
        activation_function = random.choice(
            ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'PReLU'])
        regularization = random.choice(['L1', 'L2', 'ElasticNet', 'None'])
        dropout_rate = random.uniform(0, 0.5)
        accuracy = random.uniform(0.5, 1)
        f1_score = random.uniform(0.5, 1)
        precision = random.uniform(0.5, 1)
        recall = random.uniform(0.5, 1)
        auc_roc = random.uniform(0.5, 1)
        model_name = random.choice(
            ['Logistic Regression', 'Random Forest', 'SVM', 'Neural Network', 'XGBoost'])
        model_training_data = model_training_data = self.generate_dataset_name()
        training_time = random.uniform(0.1, 10)  # Training time in hours
        training_cost = random.uniform(10, 1000)  # Training cost in dollars

        return {
            'model_name': model_name,
            'model_training_data': model_training_data,
            'learning_rate': round(learning_rate, 4),
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'optimizer': optimizer,
            'loss_function': loss_function,
            'activation_function': activation_function,
            'regularization': regularization,
            'dropout_rate': dropout_rate,
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1_score, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'auc_roc': round(auc_roc, 4),
            'training_time': round(training_time, 2),
            'training_cost': round(training_cost, 2)
        }


class ProspectInteractionGenerator:
    def __init__(self):
        self.interactions = [
            "Donor wants to donate to cancer fund.",
            "Donor wants her child to attend the program.",
            "Donor is interested in funding a scholarship.",
            "Donor would like to sponsor a research project.",
            "Donor is interested in attending a university event.",
            "Donor would like to volunteer for a university program.",
            "Donor is considering endowing a professorship.",
            "Donor wants to contribute to the university's capital campaign.",
            "Donor is interested in supporting the university's athletics program.",
            "Donor wants to fund a student exchange program."
        ]
        self.donor_classifications = [
            "VIP",
            "major_prospect",
            "new_prospect",
            "recently_engaged_prospect"
        ]
        self.urgency_phrases = [
            "requires immediate attention",
            "should be addressed soon",
            "can be prioritized in the coming weeks",
            "is not urgent, but should not be ignored"
        ]

    def generate_interaction(self):
        interaction_object = {
            "event_type": "prospect_interaction",
            "prospect_id": random.randint(1000, 9999),
            "donation_purpose": None,
            "description": None,
            "amount": None,
            "gift_category": None,
            "interaction_date": self.random_date_between_now_and_last_year(),
            "interaction": random.choice(self.interactions),
            "donor_classification": random.choice(self.donor_classifications),
            "urgency": random.choice(self.urgency_phrases)
        }

        return interaction_object

    def random_date_between_now_and_last_year(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365)
        delta = end - start
        random_days = random.randrange(delta.days)
        date = start + datetime.timedelta(days=random_days)
        return date.strftime('%Y-%m-%d')
