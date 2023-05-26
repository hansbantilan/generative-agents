import yaml
import openai

def load_params(path: str) -> dict:
    with open(path) as f:
        params = yaml.safe_load(f)
    return params

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]