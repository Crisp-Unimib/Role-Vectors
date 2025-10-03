import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    """
    Creates a hook function to compute and cache the mean activations of a specified layer during the forward pass.

    Args:
        layer (int): The index of the layer for which the mean activations are being computed.
        cache (Float[Tensor, "pos layer d_model"]): A tensor to store the mean activations.
        n_samples (int): The number of samples over which the mean is computed.
        positions (List[int]): The positions in the sequence to consider for computing the mean activations.

    Returns:
        hook_fn (function): A hook function that can be registered to a layer to compute and cache the mean activations.
    """
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    """
    Compute the mean activations of specified layers and positions in a model for a given set of instructions.

    Args:
        model: The model for which activations are to be computed.
        tokenizer: The tokenizer used to preprocess the instructions.
        instructions (List[str]): A list of instructions to be tokenized and fed into the model.
        tokenize_instructions_fn (Callable): A function to tokenize the instructions.
        block_modules (List[torch.nn.Module]): A list of model modules (layers) to hook for capturing activations.
        batch_size (int, optional): The number of instructions to process in each batch. Default is 32.
        positions (List[int], optional): The positions within each sequence to capture activations from. Default is [-1].

    Returns:
        torch.Tensor: A tensor containing the mean activations with shape (n_positions, n_layers, d_model).
    """
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations

def get_mean_diff(model, tokenizer, target_instructions, base_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    """
    Computes the mean difference in activations between harmful and harmless instructions.

    Args:
        model: The language model to be used.
        tokenizer: The tokenizer associated with the model.
        harmful_instructions (List[str]): A list of harmful instructions.
        harmless_instructions (List[str]): A list of harmless instructions.
        tokenize_instructions_fn (Callable): A function to tokenize the instructions.
        block_modules (List[torch.nn.Module]): A list of model modules to extract activations from.
        batch_size (int, optional): The batch size to use for processing instructions. Defaults to 32.
        positions (List[int], optional): The positions in the sequence to consider for activations. Defaults to [-1].

    Returns:
        torch.Tensor: The mean difference in activations between harmful and harmless instructions, 
                      with shape (n_positions, n_layers, d_model).
    """
    mean_activations_target = get_mean_activations(model, tokenizer, target_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_activations_base = get_mean_activations(model, tokenizer, base_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_target - mean_activations_base

    return mean_diff

def generate_directions(model_base: ModelBase, target_instructions, base_instructions, artifact_dir):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model_base.model, model_base.tokenizer, target_instructions, base_instructions, model_base.tokenize_instructions_fn, model_base.model_block_modules, positions=list(range(-len(model_base.eoi_toks), 0)))

    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs