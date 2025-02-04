# ConversationalFormAI

## Overview
This repository contains the implementation of **"Cost Saving: A Conversational AI Approach for Automated Form Filling."** The project explores using Conversational AI and speech recognition to automate form completion.

## Features
- **Speech-to-Text:** Uses OpenAI Whisper for transcription.
- **Decision Tree Model:** Provides structured decision-making for form processing.
- **Flask Backend:** Handles API requests and form storage.
- **LLM Integration:** Uses GPT-4 to clarify ambiguous responses.

## Installation
To set up the project locally:

```bash
# Clone the repository
git clone https://github.com/your-username/ConversationalFormAI.git
cd ConversationalFormAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run the Flask app
python app.py
