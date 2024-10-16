categories = ["non-hateful","hateful"]

OPENAI_KEY = "xxx"
TOGETHER_KEY = "xxx"

def generate_prompt(text,label):
    return f"""
    Classify the text into hateful and non-hateful, and return the answer as the corresponding label.
    text: {text}
    label: {label}""".strip()

def generate_test_prompt(text):
    return f"""
    Classify the text into hateful and non-hateful, and return the answer as the corresponding label.
    text: {text}
    label: """.strip()
