# """
# A sample Hello World server.
# """
# import os

# from flask import Flask, render_template

# # pylint: disable=C0103
# app = Flask(__name__)


# @app.route('/')
# def hello():
#     """Return a friendly HTTP greeting."""
#     message = "It's running!"

#     """Get Cloud Run environment variables."""
#     service = os.environ.get('K_SERVICE', 'Unknown service')
#     revision = os.environ.get('K_REVISION', 'Unknown revision')

#     return render_template('index.html',
#         message=message,
#         Service=service,
#         Revision=revision)

# if __name__ == '__main__':
#     server_port = os.environ.get('PORT', '8080')
#     app.run(debug=False, port=server_port, host='0.0.0.0')


import vertexai
from vertexai.language_models import TextGenerationModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# Initialize Vertex AI
vertexai.init(project="your-project", location="us-central1")

# Define parameters for text generation
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 1
}

# Load the pre-trained model
model = TextGenerationModel.from_pretrained("text-bison@001")

# Create Flask app
app = Flask(__name__)
CORS(app)

vertexai.init(project="career-path-navigator", location="us-central1")


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, 'text' key is required."}), 400

    # Extract user input
    user_question = data['text']
    
    # Generate a prompt for the model
    prompt = f"""
Career counseling session:

You are a career counselor providing detailed advice on various aspects of a professional's career. Here is a sample professional profile:

Q: What is the market value and salary range for this role in the industry?
A: The market value and salary range can vary depending on factors such as location, experience, and industry demand. Typically, provide details relevant to india.
Q: What relevant education or standalone courses would benefit someone pursuing this career?
A: Relevant education might include degrees in Data Science, Computer Science, or related fields. Standalone courses and certifications in data science, machine learning, and cloud technologies from platforms like Coursera, Udacity, or edX can also be highly beneficial.

Q: What skills are required for this role?
A: Essential skills include proficiency in data analysis, machine learning algorithms, programming languages such as Python or R, and experience with cloud platforms like AWS and Google Cloud. Strong problem-solving abilities and communication skills are also important.

Q: What does a typical job description for this role include?
A: A typical job description might include responsibilities such as analyzing complex data sets, developing machine learning models, implementing data-driven solutions, collaborating with cross-functional teams, and staying updated with the latest industry trends and technologies.

Q: {user_question}
A:
"""

    # Get the model's response
    response = model.predict(prompt, **parameters)

    # Return the response as JSON
    return jsonify({"response": response.text}), 200

if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0', debug=True)


