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

## Usage

Once the app is running, you can:

### Create a Form
- Access the home page to design a new form by entering the sender, title, and questions.

### Share the Form
- Distribute the generated URL to users. Each participant is assigned a unique session ID.

### Submit Audio Responses
- Users record their answers.
- The app transcribes audio using Whisper, evaluates transcription confidence, and—if needed—escalates unclear responses to GPT-4 for clarification.

### Review Responses
- Responses are stored in JSON format and can be viewed via the provided endpoints.

## License

This project is licensed under the [MIT License](LICENSE).
