import google.generativeai as genai

genai.configure(api_key)
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello")
    print("SUCCESS:", response.text)
except Exception as e:
    print("ERROR:", str(e))
