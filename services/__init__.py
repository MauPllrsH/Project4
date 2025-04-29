"""
External service integrations for the CSV & OpenAI Integration application.
This package contains modules for interacting with external APIs like OpenAI.
"""

# Allow direct imports from the services package
from .openai_service import call_openai_api, generate_data_description
