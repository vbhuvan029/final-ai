from google import genai

try:
    client = genai.Client(api_key)
    response = client.models.generate_content(model='gemini-2.5-flash', contents='Hello')
    print("SUCCESS 2.5:", response.text)
except Exception as e:
    print("ERROR 2.5:", str(e))
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents='Hello')
        print("SUCCESS 2.0:", response.text)
    except Exception as e2:
        print("ERROR 2.0:", str(e2))
