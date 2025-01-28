import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from byte_tokenizer import encode, decode, get_data_from_file, prepare_text, get_batch
from safetensors.torch import save_model, load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def train(
    model: Transformer,
    data: List[List[int]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_path: str
) -> None:
    torch.autograd.set_detect_anomaly(True)
    """
    Trains the model on the given data.

    Args:
        model (Transformer): The transformer model to be trained.
        data (List[List[int]]): The training data.
        epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        save_path (str): The path to save the trained model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require gradients")
    
    for epoch in range(epochs):
        total_loss = 0
        temperature = 0.666
        for i in range(0, len(data), batch_size):
            optimizer.zero_grad()
            batch, target = get_batch(data, i, batch_size)
            #print("Batch Tokens : " , batch)
            target = target.to(torch.long).squeeze()
            logits = model.forward(batch)
            next_token = sample(logits, temperature)
            print(f"Next token : {next_token}")
            #print(f"Model outputs: {logits}")
            #print(f"Logits shape: {logits.shape}, Target shape: {target.shape}")
            #print(logits.view(-1, logits.size(-1)), target.view(-1))
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            print(f"Loss before backward: {loss.item()}")
            loss.backward()
            print(loss)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    print(f"Gradient norm for {name}: {grad_norm}")

            for name, param in model.named_parameters():
                print(f"Parameter {name} before step: {param.data}")


            optimizer.step()

            for name, param in model.named_parameters():
                print(f"Parameter {name} after step: {param.data}")
            
            total_loss += loss.item()
            break
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")
    
    save_model(model, save_path)
    print(f"Model saved to {save_path}")


def main(
    ckpt_path: str,
    config: str,
    data_file: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.01,
    save_path: str = "model.safetensors"
) -> None:
    """
    Main function to load the model and perform training.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        data_file (str): Path to the file containing training data.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        save_path (str, optional): Path to save the trained model. Defaults to "model.safetensors".
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("gloo")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    model = Transformer(args)
    checkpoint_path = os.path.join(ckpt_path, f"model.safetensors")
    if os.path.exists(checkpoint_path):
        load_model(model, checkpoint_path)
        print("Loaded model from checkpoint.")
    else:
        model.apply(model._init_weights)
        print("No checkpoint found. Initializing a random model.")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name}")

    data = get_data_from_file(data_file)
    train(model, data, epochs, batch_size, learning_rate, save_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")
    """
    Command-line interface for distributed model training.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --data-file (str): File containing training data.
        --epochs (int, optional): Number of training epochs. Defaults to 10.
        --batch-size (int, optional): Batch size for training. Defaults to 32.
        --learning-rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        --save-path (str, optional): Path to save the trained model. Defaults to "model.safetensors".
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="data/checkpoints")
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--data-file", type=str, default="data/tiny_shakespear.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--save-path", type=str, default="model.safetensors")
    args = parser.parse_args()
    main(args.ckpt_path, args.config, args.data_file, args.epochs, args.batch_size, args.learning_rate, args.save_path)