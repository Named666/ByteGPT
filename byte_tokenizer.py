import torch

# Dataset Handling
def get_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def encode(s):
    bytes_list = bytes(s, encoding='utf-8')
    bytes_list = list(bytes_list)
    return bytes_list

def decode(l):
    decoded_bytes = bytes(l).decode('utf-8', errors='replace')
    return decoded_bytes

def prepare_text(text):
    encoded_text = encode(text)
    tensor = torch.tensor(encoded_text, dtype=torch.bfloat16)
    return tensor

def get_batch(data, start_index, batch_size):
    end_index = start_index + batch_size
    x = [encode(data[i]) for i in range(start_index, end_index)]
    y = [encode(data[i + 1]) for i in range(start_index, end_index)]
    x = torch.tensor(x, dtype=torch.bfloat16, requires_grad=True)
    y = torch.tensor(y, dtype=torch.bfloat16, requires_grad=True)
    return x, y