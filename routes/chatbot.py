from flask import Blueprint, current_app, jsonify, render_template, request
from flask_login import login_required
from groq import Groq

chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.route("/chatbot")
@login_required
def chat_interface():
    return render_template("chatbot.html")


@chatbot_bp.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"response": "Please enter a valid message."})

    try:
        client = Groq(api_key=current_app.config["GROQ_API_KEY"])

        system_instruction = (
            "You are an AI Health Assistant for a web application. "
            "Provide helpful, concise, and accurate medical guidance, first aid steps, "
            "or home remedies based on the user's query. "
            "Always include a short disclaimer that this does not replace professional medical diagnosis and that the user should consult a doctor. "
            "Keep responses under 3 short paragraphs. "
            "Use simple HTML tags like <b>, <ul>, <li>, and <br> when helpful. "
            "Do not use markdown asterisks."
        )

        response = client.chat.completions.create(
            model=current_app.config["GROQ_MODEL"],
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": message},
            ],
            temperature=0.4,
            max_tokens=512,
        )

        content = (response.choices[0].message.content or "").strip()
        formatted_response = content.replace("**", "<b>").replace("\n", "<br>")

        return jsonify({"response": formatted_response})

    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        return jsonify(
            {
                "response": "I'm sorry, I am having trouble connecting to my AI brain right now. Please try again later."
            }
        )
