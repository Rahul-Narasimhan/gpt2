import torch
import torch.nn.functional as F
import tiktoken
import os

from gpt2_model import GPT, GPTConfig

log_dir = '/content/drive/MyDrive/gpt_2_wiki2M/log/'

def generate_text(
    model,
    device,
    prompt="The",
    max_new_tokens=150,
    num_return_sequences=3,
    temperature=0.7,
    top_k=20,
):
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x = x.repeat(num_return_sequences, 1)

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -model.config.block_size:]

            logits, _ = model(x_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(topk_vals, dim=-1)
                next_idx_in_topk = torch.multinomial(probs, num_samples=1)
                x_next = torch.gather(topk_idx, -1, next_idx_in_topk)
            else:
                probs = F.softmax(logits, dim=-1)
                x_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, x_next), dim=1)

    outputs = []
    for i in range(num_return_sequences):
        outputs.append(enc.decode(x[i].tolist()))
    return outputs


def load_model_for_inference(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    config_data = ckpt["config"]

    # handle both dict-style and GPTConfig-style checkpoints
    if isinstance(config_data, dict):
        config = GPTConfig(**config_data)
    else:
        config = config_data

    model = GPT(config)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"]

    # fallback for compiled-model checkpoints
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, ckpt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    checkpoint_path = os.path.join(log_dir, 'model_00099.pt')
    model, ckpt = load_model_for_inference(checkpoint_path, device)
    print(f"loaded checkpoint from step {ckpt['step']}")

    print("!!!!!!!!!!!!!! Generating output !!!!!!!!!!!!!!")
    outputs = generate_text(
    model=model,
    device=device,
    prompt="The history of science",
    max_new_tokens=600,
    num_return_sequences=5,
    temperature=0.7,
    top_k=50,
  )

    for i, text in enumerate(outputs):
        print(f"\n--- Sample {i+1} ---")
        print(text)
        print("\n[repr view]")
        print(repr(text))


if __name__ == "__main__":
    main()