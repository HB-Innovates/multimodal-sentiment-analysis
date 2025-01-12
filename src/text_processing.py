def load_data(file_path):
    # Load text data from the specified file path
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

def tokenize_text(text, tokenizer):
    # Tokenize the input text using the specified tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens

def prepare_text_data(data, tokenizer, max_length=512):
    # Prepare text data for model input
    tokenized_data = []
    for text in data:
        tokens = tokenize_text(text, tokenizer)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        tokenized_data.append(tokens)
    return tokenized_data