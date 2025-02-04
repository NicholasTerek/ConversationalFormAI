import os
import uuid
import json
import logging
import warnings
import math
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import torch
import torchaudio
import soundfile as sf

from openai import OpenAI
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_cors import CORS

from sklearn.tree import DecisionTreeClassifier
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import logging as transformers_logging

###############################################################################
# 1. INITIAL SETUP
###############################################################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings and debug messages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
transformers_logging.set_verbosity_error()

# Suppress pydub debug logging
logging.getLogger("pydub.converter").setLevel(logging.WARNING)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Whisper model and processor
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "super-secret-key"  # For flash messages
CORS(app)

# Ensure necessary directories exist
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_FORMS_FOLDER = "data/forms"
os.makedirs(DATA_FORMS_FOLDER, exist_ok=True)


###############################################################################
# 2. DECISION TREE SETUP
###############################################################################

# Example Decision Tree for Yes/No structured responses
X_train = np.array([[0], [1], [0], [1]])  # 0 = No, 1 = Yes
y_train = ["Proceed", "Stop", "Proceed", "Stop"]

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
logger.info("Decision Tree model trained successfully.")

def decide_with_tree(input_value):
    """Predicts an action based on structured input using a decision tree."""
    try:
        prediction = tree.predict([[input_value]])
        logger.info(f"Decision Tree Input: {input_value} → Prediction: {prediction[0]}")
        return prediction[0]
    except Exception as e:
        logging.error(f"Decision Tree error: {e}")
        return "Uncertain"


###############################################################################
# 3. AI MODULES - TRANSCRIPTION & UNCERTAINTY MEASUREMENT
###############################################################################

def compute_confidence_from_logprobs(logprobs, method="avg"):
    """
    Convert a list of per-token log probabilities into a single 0..1 confidence.
    method can be 'avg', 'median', or 'min'.
    """
    if not logprobs:
        return 0.0

    import numpy as np

    if method == "avg":
        lp = np.mean(logprobs)
    elif method == "median":
        lp = np.median(logprobs)
    elif method == "min":
        lp = min(logprobs)
    else:
        lp = np.mean(logprobs)

    if math.isinf(lp):
        return 0.0  # clamp -inf => confidence=0
    
    lp_shifted = lp + 14 #SHIFT VALUE

    confidence = math.exp(lp_shifted)
    return confidence

def transcribe_audio(
    file_path,
    confidence_method="avg", 
    num_beams=1, 
    skip_special=False
):
    """
    Convert .webm to .wav with pydub, then transcribe with local Whisper.
    
    :param file_path: path to the .webm file
    :param confidence_method: 'avg', 'median', or 'min' for how to compute confidence
    :param num_beams: if > 1, run beam search for N-best decoding
    :param skip_special: if True, skip special tokens (like <|startoftranscript|>)
                         when collecting log probabilities
    :return: (transcribed_text, confidence) for the best sequence
    """
    try:
        # 1) Convert webm -> wav
        temp_wav = file_path.replace(".webm", "") + "_converted.wav"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = AudioSegment.from_file(file_path, format="webm")
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # Normalize to [-1, 1]
        samples = samples / (1 << (8 * audio.sample_width - 1))

        # Save as WAV
        sf.write(temp_wav, samples, 16000, 'PCM_16')
        logger.info(f"Converted audio saved: {temp_wav}")

        # 2) Load the WAV via soundfile
        speech_array, sr = sf.read(temp_wav)
        speech_tensor = torch.from_numpy(speech_array).unsqueeze(0)

        # Cleanup
        try:
            os.remove(temp_wav)
        except Exception:
            pass

        # Basic check
        speech_tensor = speech_tensor.squeeze()
        if speech_tensor.ndim == 0:
            raise ValueError("Empty audio file")

        # 3) Preprocess
        inputs = processor(
            speech_tensor, 
            sampling_rate=sr,
            return_tensors="pt",
            language="en",
            task="transcribe"  
        )
        for k,v in inputs.items():
            inputs[k] = v.to(device)

        # 4) Generate (single or multi-beam)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=75,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams
            )

        sequences = generated.sequences
        scores_list = generated.scores  # list of length seq_len-1, each shape=[batch_size * num_return_sequences, vocab_size]

        # If num_beams=1 => single best. If >1 => N-best
        if num_beams == 1:
            # Single sequence
            token_ids = sequences[0]
            logprobs = []
            for step, step_logits in enumerate(scores_list):
                step_lp = torch.log_softmax(step_logits, dim=-1)
                chosen_id = token_ids[step+1].item()
                if skip_special and chosen_id in processor.tokenizer.all_special_ids:
                    continue

                val = step_lp[0, chosen_id].item()
                print(f"Token {step}: id={chosen_id}, logprob={val:.4f}, text='{processor.tokenizer.decode([chosen_id])}'")
                logprobs.append(val)


            # Compute confidence
            confidence = compute_confidence_from_logprobs(logprobs, method=confidence_method)
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0]

            # Logging
            logger.info(f"Used single-sequence, method={confidence_method}, skip_special={skip_special}")
            logger.info(f"Transcription text: {text}")
            logger.info(f"Confidence: {confidence:.3f}")

            return text, confidence
        else:
            # MULTI-BEAM approach
            num_hyp = sequences.shape[0]  # should be = num_beams
            beam_logprobs = [[] for _ in range(num_hyp)]

            for step, step_logits in enumerate(scores_list):
                step_lp = torch.log_softmax(step_logits, dim=-1)  # shape [num_hyp, vocab_size]
                for b in range(num_hyp):
                    token_id = sequences[b, step+1].item()
                    # skip special if desired
                    if skip_special and token_id in processor.tokenizer.all_special_ids:
                        continue
                    beam_logprobs[b].append(step_lp[b, token_id].item())

            beam_confidences = []
            for b in range(num_hyp):
                c = compute_confidence_from_logprobs(beam_logprobs[b], method=confidence_method)
                beam_confidences.append(c)

            # pick best beam
            best_beam_idx = max(range(num_hyp), key=lambda i: beam_confidences[i])
            best_conf = beam_confidences[best_beam_idx]
            best_text = processor.batch_decode(
                sequences[best_beam_idx:best_beam_idx+1],
                skip_special_tokens=True
            )[0]

            # optional second-best margin
            sorted_beams = sorted(range(num_hyp), key=lambda i: beam_confidences[i], reverse=True)
            if len(sorted_beams) >= 2:
                second_idx = sorted_beams[1]
                second_conf = beam_confidences[second_idx]
                delta = best_conf - second_conf
                logger.info(f"Beam search best={best_conf:.3f}, 2nd={second_conf:.3f}, delta={delta:.3f}")

            logger.info(f"Used multi-beam={num_beams}, method={confidence_method}, skip_special={skip_special}")
            logger.info(f"Best beam text: {best_text}")
            logger.info(f"Best beam confidence: {best_conf:.3f}")

            return best_text, best_conf

    except Exception as e:
        logger.error(f"Critical Error in transcription: {str(e)}")
        return None, None


def calculate_uncertainty(avg_confidence):
    """Estimates uncertainty based on the final confidence score (0..1)."""
    if avg_confidence is None:
        return 1.0
    uncertainty = 1.0 - avg_confidence
    logger.info(f"Uncertainty Score: {uncertainty:.4f}")
    return uncertainty


###############################################################################
# 3. CREATE AND SHARE FORM
###############################################################################

@app.route('/')
def index():
    """Show a form to create a new shared audio form."""
    return render_template("create_form.html")

@app.route('/create_form', methods=['POST'])
def create_form():
    """Creates a new form with the provided sender, title, and questions."""
    try:
        sender = request.form.get('sender', '').strip()
        title = request.form.get('title', '').strip()
        raw_questions = request.form.get('questions', '').strip()

        if not sender or not title or not raw_questions:
            flash('Please fill in all fields', 'error')
            return render_template("create_form.html")

        questions_list = [q.strip() for q in raw_questions.split('\n') if q.strip()]
        if not questions_list:
            flash('Please add at least one question', 'error')
            return render_template("create_form.html")

        form_data = {
            "sender": sender,
            "title": title,
            "questions": questions_list,
            "created_at": datetime.now().isoformat(),
            "sub_sessions": {}
        }

        form_id = str(uuid.uuid4())
        form_path = os.path.join('data', 'forms', f'{form_id}.json')
        with open(form_path, 'w', encoding='utf-8') as f:
            json.dump(form_data, f, ensure_ascii=False, indent=2)

        share_url = request.host_url.rstrip('/') + url_for('join_form', form_id=form_id)
        return render_template("share_link.html", share_url=share_url, form_id=form_id)

    except Exception as e:
        app.logger.error(f"Form creation error: {str(e)}")
        flash('An error occurred while creating the form. Please try again.', 'error')
        return render_template("create_form.html")


###############################################################################
# 4. JOIN THE FORM (AUTO-GENERATE A USER ID) & ANSWER A QUESTION
###############################################################################

@app.route('/form/<form_id>')
def join_form(form_id):
    """
    Main entry to the form: If a user_id isn't provided, generate one
    and redirect to ?u=<user_id>.
    """
    form_path = os.path.join('data', 'forms', f'{form_id}.json')
    if not os.path.exists(form_path):
        return render_template("error.html",
                               title="Form Not Found",
                               message="This form does not exist or has been deleted.")

    user_id = request.args.get('u')
    if not user_id:
        new_user_id = str(uuid.uuid4())
        return redirect(url_for('join_form', form_id=form_id, u=new_user_id))

    return render_answer_page(form_id, user_id)

def render_answer_page(form_id, user_id):
    """
    Loads the form JSON, ensures sub_sessions[user_id] exists,
    and renders the next question or a 'done' message.
    """
    form_path = os.path.join('data', 'forms', f'{form_id}.json')
    with open(form_path, 'r', encoding='utf-8') as f:
        form_data = json.load(f)

    if "sub_sessions" not in form_data:
        form_data["sub_sessions"] = {}

    if user_id not in form_data["sub_sessions"]:
        form_data["sub_sessions"][user_id] = {
            "current_index": 0,
            "answers": []
        }
        with open(form_path, 'w', encoding='utf-8') as f_out:
            json.dump(form_data, f_out, ensure_ascii=False, indent=2)

    sub_session = form_data["sub_sessions"][user_id]
    questions = form_data["questions"]
    total_questions = len(questions)
    current_index = sub_session["current_index"]

    if current_index >= total_questions:
        is_done = True
        question_text = ""
    else:
        is_done = False
        question_text = questions[current_index]

    personal_link = request.host_url.rstrip('/') + url_for('join_form', form_id=form_id) + f"?u={user_id}"

    return render_template("answer_info.html",
                           form_data=form_data,
                           form_id=form_id,
                           user_id=user_id,
                           question=question_text,
                           total_questions=total_questions,
                           current_index=current_index,
                           personal_link=personal_link,
                           is_done=is_done)

@app.route('/form/<form_id>/responses')
def show_responses(form_id):
    """
    Loads the form JSON and renders a page that displays all responses.
    """
    form_path = os.path.join('data', 'forms', f'{form_id}.json')
    if not os.path.exists(form_path):
        return render_template("error.html",
                               title="Form Not Found",
                               message="This form does not exist or has been deleted.")
    with open(form_path, 'r', encoding='utf-8') as f:
        form_data = json.load(f)
    return render_template("show_responses.html", form_data=form_data, form_id=form_id)


###############################################################################
# 5. SUBMIT AUDIO (PER USER SUB-SESSION)
###############################################################################

@app.route('/form/<form_id>/user/<user_id>/submit-audio', methods=['POST'])
def submit_audio_response(form_id, user_id):
    """
    Handles audio transcription, applies Decision Tree first, then LLM if uncertainty is high.
    """
    form_path = os.path.join('data', 'forms', f'{form_id}.json')
    if not os.path.exists(form_path):
        return jsonify({"error": "Form not found"}), 404

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "Invalid audio file"}), 400

    with open(form_path, 'r', encoding='utf-8') as f:
        form_data = json.load(f)

    if user_id not in form_data["sub_sessions"]:
        return jsonify({"error": "User session not found"}), 400

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.webm")
    file.save(file_path)

    # Example: we can pass optional parameters to transcribe_audio
    # e.g., transcribe_audio(file_path, "median", num_beams=3, skip_special=True)
    transcription, confidence = transcribe_audio(file_path,
                                                confidence_method="avg",
                                                num_beams=1,
                                                skip_special=True)

    os.unlink(file_path)

    if not transcription:
        return jsonify({"error": "Failed to transcribe audio"}), 500

    # Calculate uncertainty
    uncertainty = calculate_uncertainty(confidence)
    logger.info(f"uncertainty = {uncertainty}")

    LOW_CONF_THRESHOLD = 0.3
    VERY_LOW_CONF_THRESHOLD = 0.7

    if uncertainty < LOW_CONF_THRESHOLD:
        # Accept
        form_data["sub_sessions"][user_id]["answers"].append(transcription)
        form_data["sub_sessions"][user_id]["current_index"] += 1
        status_msg = "Accepted response"
    elif uncertainty < VERY_LOW_CONF_THRESHOLD:
        # Escalate to GPT
        llm_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Clarify ambiguous responses in a form-filling context."},
                {"role": "user", "content": f"User said: {transcription}. Please clarify what they meant."}
            ]
        )
        clarified_text = llm_response.choices[0].message.content
        form_data["sub_sessions"][user_id]["answers"].append(clarified_text)
        status_msg = "Escalated to GPT-4"
    else:
        # Re-ask user
        return jsonify({
            "message": "Confidence too low. Please re-record your answer.",
            "nextUrl": None
        }), 200

    with open(form_path, 'w', encoding='utf-8') as f_out:
        json.dump(form_data, f_out, indent=2)

    next_url = url_for('join_form', form_id=form_id, u=user_id, _external=True)
    return jsonify({"message": status_msg, "nextUrl": next_url}), 200


###############################################################################
# 6. START APP
###############################################################################

if __name__ == '__main__':
    app.run(debug=True)
