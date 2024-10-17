OPENAI_KEY = "xxx"
TOGETHER_KEY = "xxx"

categories = ["normal-speech","hate-speech"]

def generate_test_prompt_with_examples(text,examples):
    return f"""
    Classify the text into hate-speech and normal-speech, and return the answer as the corresponding label.
    example 1:
    text: {examples[0]["text"]}
    label: {examples[0]["label"]}
    
    example 2:
    text: {examples[1]["text"]}
    label: {examples[1]["label"]}
    
    example 3:
    text: {examples[2]["text"]}
    label: {examples[2]["label"]}
    
    example 4:
    text: {examples[3]["text"]}
    label: {examples[3]["label"]}
    
    text: {text}
    label: """.strip()

def generate_prompt(text,label):
    return f"""
    Classify the text into hate-speech and normal-speech, and return the answer as the corresponding label.
    text: {text}
    label: {label}""".strip()

def generate_test_prompt(text):
    return f"""
    Classify the text into hate-speech and normal-speech, and return the answer as the corresponding label.
    text: {text}
    label: """.strip()
