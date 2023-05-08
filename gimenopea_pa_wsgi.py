import sys
import os

# Add your project directory to the sys.path
project_home = u'/home/gimenopea/capstone_demo'

# Load environment variables from .env file
from dotenv import load_dotenv

project_folder = os.path.expanduser('~/capstone_demo')  # Adjust as appropriate
load_dotenv(os.path.join(project_folder, '.env'))

if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Import Flask app, but need to call it "application" for WSGI to work
from app import app as application  # noqa
