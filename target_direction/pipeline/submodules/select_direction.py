import json
import torch
import math
import matplotlib.pyplot as plt
import os
import asyncio

from typing import List, Optional
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from einops import rearrange
from pathlib import Path

from pipeline.dataset.load_datasets import load_dataset
from pipeline.dataset.generator import Generator
from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook

def likelihood_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    answer_toks: list,  
    epsilon: Float = 1e-8,
):
    """
    Compute the log odds score between the summed probabilities of correct and incorrect answer tokens.
    This function converts the input logits to float64 and focuses on the logits
    corresponding to the last token of each sequence. It then applies a softmax to
    obtain a probability distribution over the vocabulary and sums the probabilities
    for the correct tokens (as specified by the first element in answer_toks) and the
    incorrect tokens (as specified by the second element in answer_toks). The final
    score is computed as the logarithm of the sum of correct probabilities (with a small
    epsilon added for numerical stability) minus the logarithm of the sum of incorrect
    probabilities (also with epsilon added).
    Parameters
    ----------
    logits : torch.Tensor
        A tensor of shape (batch, seq, d_vocab_out) representing the model's output logits.
    answer_toks : list
        A list containing two elements:
          - The first element corresponds to the indices for the correct tokens.
          - The second element corresponds to the indices for the incorrect tokens.
    epsilon : float, optional
        A small constant (default: 1e-8) added to the probabilities to prevent log(0)
        during computation.
    Returns
    -------
    torch.Tensor
        A tensor of shape (batch,) containing the computed log odds scores.
    """
    logits = logits.to(torch.float64)

    logits = logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    correct_probs = probs[:, answer_toks[0]].sum(dim=-1)

    incorrect_probs = probs[:, answer_toks[1]].sum(dim=-1)

    return torch.log(correct_probs + epsilon) - torch.log(incorrect_probs + epsilon)

def performance_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    answer_toks: list,  
):
    """
    Calculates a performance score based on the model's output logits and a set of answer tokens.
    Args:
        logits (Tensor): A tensor of shape (batch, seq, d_vocab_out) representing the unnormalized log probabilities.
                         The logits are first converted to float64 and then reduced to the last sequence step.
        answer_toks (list): A list where the first element is the reference answer token (expected correct token)
                            and the second element is an iterable containing candidate answer tokens.
    Returns:
        tuple: A tuple containing:
            - score (Tensor): A tensor of floats with shape (batch,) where each element is 1.0 if the selected token 
                              matches the reference answer token (answer_toks[0]), otherwise 0.0.
            - selected_token (Tensor): A tensor of the selected answer token for each example in the batch.
    """
    logits = logits.to(torch.float64)

    logits = logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    all_answer_tokens = [answer_toks[0]] + answer_toks[1]
    answer_probs = probs[:, all_answer_tokens]
    selected_token_idx = torch.argmax(answer_probs, dim=-1)
    selected_token = torch.tensor([all_answer_tokens[idx] for idx in selected_token_idx], 
                                device=logits.device)

    score = (selected_token == answer_toks[0]).float()

    return score, selected_token

# def get_direction_scores(
#     model,
#     instructions,
#     tokenize_instructions_fn,
#     answer_toks,
#     fwd_pre_hooks=[],
#     fwd_hooks=[],
#     batch_size=8,
#     debug=False 
# ):
#     """
#     Computes likelihood and performance scores for a batch of instructions by forwarding 
#     them through the given model in segments, applying tokenization and hook functions as needed.
#     Parameters:
#         model (torch.nn.Module): The model used for computing logits. It must have a 'device' attribute.
#         instructions (list[dict]): A list of dictionaries where each dictionary represents an instruction.
#             Each dictionary should include at least the following keys:
#                 - "instruction": The input text to be tokenized and processed.
#                 - "target_score": A string ("A", "B", "C", or "D") indicating which answer token is the target.
#                 - "dataset": An identifier for the source dataset.
#         tokenize_instructions_fn (callable): A function that takes a list of instruction strings and returns
#             a tokenized object with attributes like 'input_ids' and 'attention_mask'.
#         answer_toks (list): A list containing answer token definitions. Each element is assumed to be a tuple 
#             where the first element corresponds to the token id for that answer option.
#         fwd_pre_hooks (list, optional): A list of pre-forward hook functions to be applied during model forwarding.
#             Defaults to an empty list.
#         fwd_hooks (list, optional): A list of forward hook functions to be applied during model forwarding.
#             Defaults to an empty list.
#         batch_size (int, optional): The number of instructions to process in a single batch. Defaults to 8.
#         debug (bool, optional): If True, enables printing of memory profiles for debugging purposes.
#             Defaults to False.
#     Returns:
#         tuple:
#             - likelihood_scores (torch.Tensor): A tensor containing the likelihood scores for each instruction.
#             - performance_scores (torch.Tensor): A tensor containing the performance scores for each instruction.
#             - results (list[dict]): A list of result dictionaries, one per instruction, each including:
#                   * "instruction": The original instruction text.
#                   * "target_answer": The target answer as specified in the instruction.
#                   * "score": The performance score (converted to float).
#                   * "selected_answer": The answer option selected from a mapping of token ids to letter labels.
#                   * "dataset": The dataset identifier from the input instruction.
#     Notes:
#         - The function leverages auxiliary functions such as 'add_hooks', 'likelihood_score', and 
#           'performance_score', which should be defined elsewhere in the codebase.
#         - Memory management is handled explicitly using 'torch.cuda.empty_cache()' and deletion 
#           of intermediary objects to help mitigate GPU memory usage during processing.
#     """
#     likelihood_scores = torch.zeros(len(instructions), device=model.device)
#     performance_scores = torch.zeros(len(instructions), device=model.device)
#     results = []
    
#     token_to_answer = {
#         answer_toks[0][0]: 'A',
#         answer_toks[1][0]: 'B', 
#         answer_toks[2][0]: 'C',
#         answer_toks[3][0]: 'D'
#     }
    
#     def profile_memory(stage):
#         if debug:
#             print(f"\n[Memory Profile] {stage}")
#             print(torch.cuda.memory_summary(device=model.device, abbreviated=True))
    
#     for i in range(0, len(instructions), batch_size):
#         torch.cuda.empty_cache()  
#         profile_memory("Start of batch")
        
#         batch_instructions = instructions[i:i+batch_size]
#         batch_answer_seqs = []
        
#         for instruction in batch_instructions:
#             if instruction["target_score"] == "A":
#                 answer_seq = [answer_toks[0][0], [x[0] for x in answer_toks[1:]]]
#             elif instruction["target_score"] == "B":
#                 answer_seq = [answer_toks[1][0], [answer_toks[0][0]] + [x[0] for x in answer_toks[2:]]]
#             elif instruction["target_score"] == "C":
#                 answer_seq = [answer_toks[2][0], [answer_toks[0][0]] + [answer_toks[1][0]] + [answer_toks[3][0]]]
#             elif instruction["target_score"] == "D":
#                 answer_seq = [answer_toks[3][0], [answer_toks[0][0]] + [answer_toks[1][0]] + [answer_toks[2][0]]]
#             batch_answer_seqs.append(answer_seq)
        
#         instructions_batch = [inst["instruction"] for inst in batch_instructions]
#         tokenized = tokenize_instructions_fn(instructions=instructions_batch)
#         profile_memory("After tokenization")
        
#         input_ids = tokenized.input_ids.to(model.device)
#         attention_mask = tokenized.attention_mask.to(model.device)
        
#         with torch.no_grad():
#             with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
#                 logits = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                 ).logits
#         profile_memory("After model forward")
        
#         del tokenized, input_ids, attention_mask
#         torch.cuda.empty_cache()
#         profile_memory("After freeing tokenized data")
        
#         for j, answer_seq in enumerate(batch_answer_seqs):
#             likelihood_scores[i+j] = likelihood_score(logits[j:j+1], answer_seq)
#             p_score, s_token = performance_score(logits[j:j+1], answer_seq)
#             performance_scores[i+j] = p_score

#             results.append({
#                 "instruction": batch_instructions[j]["instruction"],
#                 "target_answer": batch_instructions[j]["target_score"],
#                 "score": float(p_score.cpu()),
#                 "selected_answer": token_to_answer[int(s_token.cpu())],
#                 "dataset": batch_instructions[j]["dataset"],
#             })
#         profile_memory("After processing batch results")
        
#         del logits
#         torch.cuda.empty_cache()
#         profile_memory("End of batch")
    
#     likelihood_scores = likelihood_scores.cpu()
#     performance_scores = performance_scores.cpu()
#     profile_memory("After moving final scores to CPU")
    
#     return likelihood_scores, performance_scores, results

def get_direction_scores(
    model_base,
    instructions,
    answer_toks,
    cfg,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=8,
    debug=False
):
    torch.cuda.empty_cache()
    
    performance_scores = torch.zeros(len(instructions), device=model_base.model.device)
    results = []
    
    token_to_answer = {
        answer_toks[0][0]: 'A',
        answer_toks[1][0]: 'B', 
        answer_toks[2][0]: 'C',
        answer_toks[3][0]: 'D'
    }
    
    def profile_memory(stage):
        if debug:
            print(f"\n[Memory Profile] {stage}")
            print(torch.cuda.memory_summary(device=model_base.model.device, abbreviated=True))
    
    for i in range(0, len(instructions), batch_size):
        torch.cuda.empty_cache()  
        profile_memory("Start of batch")
        
        batch_instructions = instructions[i:i+batch_size]
        batch_answer_seqs = []
        
        
        # Generate completions for the current batch only
        batch_completions = model_base.generate_completions(
            batch_instructions, 
            fwd_pre_hooks=fwd_pre_hooks, 
            fwd_hooks=fwd_hooks, 
            batch_size=batch_size,
            max_new_tokens=cfg.max_new_tokens_reasoning
        )
        
        for j, completion in enumerate(batch_completions):
            p_score = 0.0
            s_token = "N/A"

            performance_scores[i+j] = p_score

            results.append({
                "instruction": batch_instructions[j]["instruction"],
                "target_answer": batch_instructions[j]["target_score"],
                "score": float(p_score),
                "selected_answer": s_token,
                "completion": completion,
                "dataset": batch_instructions[j]["dataset"],
            })
        profile_memory("After processing batch results")
        
        torch.cuda.empty_cache()
        profile_memory("End of batch")
    
    performance_scores = performance_scores.cpu()
    profile_memory("After moving final scores to CPU")
    
    return performance_scores, results

def plot_scores(
    scores: Float[Tensor, 'n_pos n_layer'],
    baseline_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    """
    Plots performance scores for tokens across multiple layers and saves the generated figure.
    Parameters:
        scores (Float[Tensor, 'n_pos n_layer']):
            A tensor of shape (n_pos, n_layer) containing performance scores for different tokens.
        baseline_score (Optional[float]):
            An optional baseline value to be displayed as a horizontal dashed line in the plot. If provided,
            the baseline is annotated on the plot.
        token_labels (List[str]):
            A list of token labels corresponding to the positions in the `scores` tensor.
        title (str):
            The title of the plot.
        artifact_dir (str):
            The directory in which the plot will be saved. Necessary directories will be created if they do not exist.
        artifact_name (str):
            The base name (without file extension) to use when saving the plot image.
    Returns:
        None
    Side Effects:
        - Generates a matplotlib plot showing performance scores across layers.
        - Saves the plot as a PNG file at the specified location.
    """
    n_pos, n_layer = scores.shape

    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_score is not None:
        ax.axhline(y=baseline_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Performance score')
    ax.legend(title='Position source of direction', loc='lower left')

    full_path = os.path.join(artifact_dir, os.path.dirname(artifact_name))
    os.makedirs(full_path, exist_ok=True)

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

def filter_fn(
              ablation_score, 
              steering_score, 
              layer, n_layer, 
              induce_performance_threshold=None, prune_layer_percentage=0.20) -> bool:
    """
            Checks whether a given configuration should be filtered based on the provided scores,
            layer index, and optional threshold criteria.

            Parameters:
                ablation_score (float): The ablation score used to determine if filtering is needed.
                steering_score (float): The steering score used to determine if filtering is needed.
                layer (int): The current layer index.
                n_layer (int): The total number of layers.
                induce_performance_threshold (Optional[float]): An optional performance threshold. If provided and
                    the steering_score is below this threshold, the configuration is filtered.
                prune_layer_percentage (float, optional): The fraction of the final layers that are automatically
                    filtered. If the current layer is in the last 'prune_layer_percentage' portion of layers,
                    the configuration is filtered. Defaults to 0.20.

            Returns:
                bool: True if any of the following conditions are met:
                      - Either ablation_score or steering_score is NaN.
                      - The current layer is within the final prune_layer_percentage of layers.
                      - The steering_score is below the induce_performance_threshold (if specified).
                      Otherwise, returns False.
            """
    if math.isnan(ablation_score) or math.isnan(steering_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if induce_performance_threshold is not None and steering_score < induce_performance_threshold:
        return True
    return False

def save_performance_results(results, prefix="baseline", artifact_dir=None, pos=None, layer=None):
    """
    Save performance results to a JSON file with a timestamp.
    This function saves a dictionary of performance results into a JSON file. If the prefix
    is "baseline", the file is saved directly under the given artifact directory with the name
    'results_baseline.json'. Otherwise, the function uses the provided 'pos' and 'layer' values
    to create a nested directory structure and names the file accordingly as
    'results_<prefix>_<pos>_<layer>.json'. The necessary directories are created if they do not exist.
    Parameters:
        results (dict):
            The performance results to be saved.
        prefix (str, optional):
            A prefix for the filename. Defaults to "baseline". Determines the naming convention.
        artifact_dir (str or Path):
            The directory path where the results file will be saved.
        pos (any, optional):
            A positional identifier used in the filename and directory structure when prefix is not "baseline".
        layer (any, optional):
            A layer identifier used in the filename and directory structure when prefix is not "baseline".
    Raises:
        Exception:
            If any error occurs during the process of saving the results, it is caught and printed.
    """
    artifact_dir = Path(artifact_dir)
    
    if prefix == "baseline":
        filename = artifact_dir / f"results_{prefix}.json"
    else:
        filename = artifact_dir / str(pos) / str(layer) / f"results_{prefix}_{pos}_{layer}.json"
        filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")

def select_direction(
    model_base: ModelBase,
    target_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    cfg,
    artifact_dir,
    induce_performance_threshold=0.0, 
    prune_layer_percentage=0.2, 
    batch_size=8,
):
    """
    Selects and returns the best candidate direction based on model performance metrics.
    This function evaluates candidate directions for modifying the model's behavior given a set of target instructions.
    It first computes the baseline performance and likelihood scores using the unmodified model, then applies a
    "steering" procedure (by adding the candidate direction to intermediate model activations) to obtain modified scores.
    If the configuration coefficient (cfg.coeff) equals 1.0, an ablation procedure is additionally performed by removing
    the candidate direction's contribution. The function then filters and sorts candidate directions based on their
    performance scores and selects the direction with the highest score that passes the filtering criteria.
    Along the way, performance results and score plots are saved to the specified artifact directory.
    Parameters:
        model_base (ModelBase): An object containing the model, its components (e.g., tokenizer, model block and attention modules),
                                and helper functions (e.g., tokenize_instructions_fn and answer_toks) required for score evaluation.
        target_instructions: The instructions or prompts used to evaluate model performance and likelihood.
        candidate_directions (Tensor of shape [n_pos, n_layer, d_model]): Tensor containing candidate direction vectors; each
                                candidate is associated with a position and a layer.
        cfg: Configuration object with attributes:
             - coeff: A scalar coefficient used to weight or modify the candidate direction during evaluation.
             - role: A string used for labeling plots and output files.
             - test: A string indicating the test split or dataset identifier.
        artifact_dir (str): Directory path where performance metrics and plot artifacts are saved.
        induce_performance_threshold (float, optional): Initial threshold for filtering directions based on performance.
                                                        Although provided as a parameter, it is reset to the baseline performance score.
                                                        Defaults to 0.0.
        prune_layer_percentage (float, optional): Fraction of the network's layers (starting from the last) to be discarded
                                                    during candidate selection. Defaults to 0.2.
        batch_size (int, optional): Batch size for evaluating the directions via model forward passes. Defaults to 8.
    Returns:
        tuple: A tuple (pos, layer, direction) where:
               - pos (int): The selected position (index in candidate_directions) of the best candidate.
               - layer (int): The selected layer index associated with the best candidate.
               - direction (Tensor): The candidate direction vector corresponding to the selected (pos, layer).
    Raises:
        AssertionError: If all candidate directions are filtered out, indicating that no valid candidate meets the
                        performance threshold criteria.
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, _ = candidate_directions.shape
    
    # Get baseline scores with single forward pass
    baseline_performance_scores, result_performance_baseline = get_direction_scores(
        model_base,
        target_instructions,
        model_base.answer_toks,
        cfg,
        batch_size=batch_size
    )

    save_performance_results(result_performance_baseline, artifact_dir=artifact_dir, prefix="baseline")
    baseline_performance_score = baseline_performance_scores.mean().item()

    induce_performance_threshold = baseline_performance_scores.mean().item()
    print(f"Inducing performance threshold: {induce_performance_threshold:.4f}")
    print(f"Baseline performance score: {baseline_performance_score:.4f}")
    
    steering_performance_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    ablation_performance_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)

    
    start_layer = n_layer // 2
    source_pos = cfg.pos  # Use the single position specified in cfg.pos

    for source_layer in tqdm(range(start_layer, n_layer), desc=f"Computing likelihood addition for source position {source_pos}"):
            torch.cuda.empty_cache()
            performance_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(cfg.coeff)

            fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_activation_addition_input_pre_hook(vector=performance_vector, coeff=coeff))]
            fwd_hooks = []

            performance_scores, result_performance_addition = get_direction_scores(
                model_base,
                target_instructions,
                model_base.answer_toks,
                cfg,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )
            save_performance_results(result_performance_addition, artifact_dir=artifact_dir, prefix="addition", pos=source_pos, layer=source_layer)
            steering_performance_scores[source_pos, source_layer] = performance_scores.mean().item()
    
    calc_ablation = (cfg.coeff == 1.0)
    if calc_ablation:
        for source_layer in tqdm(range(start_layer, n_layer), desc=f"Computing likelihood addition for source position {source_pos}"):

                performance_vector = candidate_directions[source_pos, source_layer]

                fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=performance_vector)) for layer in range(model_base.model.config.num_hidden_layers)]
                fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=performance_vector)) for layer in range(model_base.model.config.num_hidden_layers)]
                fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=performance_vector)) for layer in range(model_base.model.config.num_hidden_layers)]

                performance_scores, result_performance_ablation = get_direction_scores(
                    model_base,
                    target_instructions,
                    model_base.answer_toks,
                    cfg,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    batch_size=batch_size
                )
                save_performance_results(result_performance_ablation, artifact_dir=artifact_dir, prefix="ablation", pos=source_pos, layer=source_layer)
                ablation_performance_scores[source_pos, source_layer] = performance_scores.mean().item()
    else:
        print("Skipping ablation calculation because cfg.coeff is not 1.0")
        
        for source_pos in range(-n_pos, 0):
            for source_layer in range(n_layer):
                ablation_performance_scores[source_pos, source_layer] = baseline_performance_scores.mean().item()


    plot_scores(
        scores=steering_performance_scores,
        baseline_score=baseline_performance_scores.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title=f'Adding {cfg.role} direction on {cfg.test} split (performance)',
        artifact_dir=artifact_dir,
        artifact_name=f'actadd_scores_performance'
    )

    if calc_ablation:

        plot_scores(
            scores=ablation_performance_scores,
            baseline_score=baseline_performance_scores.mean().item(),
            token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
            title=f'Ablating {cfg.role} direction on {cfg.test} split (performance)',
            artifact_dir=artifact_dir,
            artifact_name=f'ablate_scores_performance'
        )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'steering_performance_score': steering_performance_scores[source_pos, source_layer].item(),
            })

            if calc_ablation:
                json_output_all_scores[-1].update({
                    'ablation_performance_score': ablation_performance_scores[source_pos, source_layer].item(),
                })

            steering_performance_score = steering_performance_scores[source_pos, source_layer].item()
            ablation_performance_score = ablation_performance_scores[source_pos, source_layer].item()

            sorting_score = steering_performance_score
    
            discard_direction = filter_fn(
                ablation_score=ablation_performance_score if calc_ablation else 0,
                steering_score=steering_performance_score,
                layer=source_layer,
                n_layer=n_layer,
                induce_performance_threshold=induce_performance_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'steering_performance_score': steering_performance_scores[source_pos, source_layer].item(),
            })

            if calc_ablation:
                json_output_filtered_scores[-1].update({
                    'ablation_performance_score': ablation_performance_scores[source_pos, source_layer].item(),
                })

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['steering_performance_score'], reverse=True)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    _, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}, coefficient={cfg.coeff}")
    print(f"Steering performance score: {steering_performance_scores[pos, layer]:.4f} (baseline: {baseline_performance_score:.4f})")
    if calc_ablation:
        ablation_mean = ablation_performance_scores.mean().item()
        ablation_std = ablation_performance_scores.std().item()
        print(f"Ablation performance score: {ablation_mean:.4f} ± {ablation_std:.4f} (baseline: {baseline_performance_score:.4f})")
    
    return pos, layer, candidate_directions[pos, layer]

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    """
    Compute the masked mean of a tensor along a specified dimension.
    This function calculates the mean of the elements of 'seq' along the given 'dim'. When a boolean 'mask' is provided,
    only the values corresponding to True in the mask are used in the computation. For 3-dimensional input tensors,
    the mask is automatically reshaped by adding an extra dimension. The mean is computed by summing the masked values
    and dividing by the count of selected elements. To prevent division by zero, the denominator is clamped to a minimum
    value of 1e-3, and any output corresponding to positions with a zero count is set to zero.
    Parameters:
        seq (torch.Tensor): The input tensor whose values are to be averaged.
        mask (Optional[torch.Tensor], optional): A boolean tensor of the same shape as 'seq' (or one dimension less for 3D tensors)
            indicating which values should be considered in the mean calculation. If None, the function returns the mean of all values.
        dim (int, optional): The dimension along which the mean is computed. Defaults to 1.
        keepdim (bool, optional): Whether to retain the reduced dimension with size 1 in the output.
            Defaults to False.
    Returns:
        torch.Tensor: A tensor containing the masked mean values computed along the specified dimension. If no elements are selected
        by the mask at a given position, the corresponding output value will be zero.
    """
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean
    

async def retry_test_response(generator, response, baseline, cfg, max_retries=3):
    """
    Retry test response asynchronously until a valid result is obtained or the maximum number of retries is reached.

    This function repeatedly calls the asynchronous function `test_response` with the provided parameters.
    If a valid result is returned (indicated by the first element of the result tuple not being None),
    it immediately returns that result. Otherwise, it continues retrying up to `max_retries` times.
    If all attempts fail, it returns the tuple ("Error", "Error").

    Parameters:
        generator: The generator or callable used to produce responses.
        response: The response input to be tested.
        baseline: Baseline data for comparison against generated responses.
        cfg: Configuration settings to control the testing process.
        max_retries (int, optional): Maximum number of attempts to obtain a valid result. Defaults to 3.

    Returns:
        tuple: A tuple containing the valid result if obtained, or ("Error", "Error") if no valid result is found.
    """
    for _ in range(max_retries):
        result = await test_response(generator, response, baseline, cfg)
        if result[0] is not None:
            return result
    return ("Error", "Error")

async def test_directions(model_base, candidate_directions, artifact_dir, cfg):
    """
    Asynchronously tests candidate directions by generating and evaluating completions.
    This function performs the following steps:
    1. Ensures that the output artifact directory exists.
    2. Loads a test dataset and a baseline completion file (generating and caching it if it does not exist) 
        using the provided base model.
    3. For each candidate direction (indexed by source position and layer):
        - Applies an activation addition hook based on the candidate direction and a configurable coefficient.
        - Generates completions for the test dataset using the modified model forward pass.
        - If an output file already exists and contains valid completions, it skips reprocessing; otherwise,
          it marks completions with an error state.
        - In non-offline mode, retries completions marked with an error by using a separate generator and 
          a comparison with baseline completions.
        - Writes the processed completions to a JSON output file, named based on the candidate's source layer and position.
    Parameters:
         model_base: 
              An object representing the base model used for generating completions. It is expected to have:
              - A method generate_completions(dataset, fwd_pre_hooks, fwd_hooks, max_new_tokens) for generating completions.
              - An attribute model_block_modules for accessing model blocks by layer.
         candidate_directions: 
              A tensor of shape (n_pos, n_layer, _) containing candidate direction performance vectors.
         artifact_dir: 
              A string representing the directory where candidate completions output files will be stored.
         cfg:
              A configuration object containing various settings such as:
              - model_test: The test model configuration.
              - openrouter_key(): A callable/key for accessing the OpenRouter API.
              - providers_test: Settings for test API providers.
              - temperature_test: Testing temperature for model generation.
              - max_new_tokens: Maximum number of tokens to generate.
              - artifact_path(): Base artifact path.
              - role: Role identifier used in defining the baseline directory.
              - offline: A boolean flag indicating whether to run in offline mode.
              - coeff: Coefficient value used for scaling the performance vector in the hook.
    Returns:
         None
    Notes:
         - The function uses asynchronous processing to concurrently evaluate completions for all candidate directions.
         - If errors occur in the generated completions and the system is not in offline mode, it attempts to correct them 
            by retrying the generation.
         - Cached baseline completions are used to compare and validate candidate completions.
    """
    n_pos, n_layer, _ = candidate_directions.shape
    os.makedirs(artifact_dir, exist_ok=True)

    generator = Generator(cfg.model_test, cfg.openrouter_key(), cfg.providers_test, cfg.temperature_test)

    dataset = load_dataset("test")

    baseline_dir = os.path.join(cfg.artifact_path(), cfg.role, "test_direction")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_file = os.path.join(baseline_dir, "baseline_completions.json")

    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            completions_baseline = json.load(f)
    else:
        completions_baseline = model_base.generate_completions(dataset, fwd_pre_hooks=[], fwd_hooks=[], max_new_tokens=cfg.max_new_tokens)
        with open(baseline_file, "w") as f:
            json.dump(completions_baseline, f, indent=4)

    async def process_candidate(source_pos, source_layer):
        performance_vector = candidate_directions[source_pos, source_layer]
        coeff = torch.tensor(cfg.coeff)
        actadd_fwd_pre_hooks = [
            (model_base.model_block_modules[source_layer],
             get_activation_addition_input_pre_hook(vector=performance_vector, coeff=coeff))
        ]
        actadd_fwd_hooks = []

        output_file = f'{artifact_dir}/{source_layer}_{source_pos}.json'
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                old_completions = json.load(f)
            # If any completion is marked as error we may process further
            if any(comp["passed"] == "Error" or comp["completion"] == "Error" for comp in old_completions):
                if cfg.offline:
                    completions = old_completions
                else:
                    completions = old_completions
            else:
                return
        else:
            completions = model_base.generate_completions(
                            dataset, 
                            fwd_pre_hooks=actadd_fwd_pre_hooks, 
                            fwd_hooks=actadd_fwd_hooks, 
                            max_new_tokens=cfg.max_new_tokens
                          )
            for completion in completions:
                completion["passed"], completion["completion"] = "Error", "Error"

        if cfg.offline:
            for completion in completions:
                completion["passed"], completion["completion"] = "Error", "Error"
        else:
            retry_tasks = []
            indices = []
            for i, completion in enumerate(completions):
                if completion.get("passed") == "Error" or completion.get("completion") == "Error":
                    retry_tasks.append(retry_test_response(generator, completion["response"], completions_baseline[i]["response"], cfg))
                    indices.append(i)
            if retry_tasks:
                results = await asyncio.gather(*retry_tasks)
                for idx, res in zip(indices, results):
                    completions[idx]["passed"], completions[idx]["completion"] = res

        with open(output_file, "w") as f:
            json.dump(completions, f, indent=4)

    tasks = []
    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            tasks.append(process_candidate(source_pos, source_layer))
    await asyncio.gather(*tasks)
    return

async def test_response(generator, response, baseline, cfg):
    """
    Asynchronously evaluates whether a given text response is related to a specific role or domain based on a baseline reference.
    Parameters:
        generator: An asynchronous text generator with a `generate` method. It takes an instruction string and returns the generated text.
        response (str): The text that needs to be evaluated.
        baseline (str): A reference text used to compare against the response to detect near-identical or generic outputs.
        cfg: A configuration object containing at least the attribute `role`, which specifies the role or content domain for evaluation.
    Returns:
        tuple: A tuple where the first element is either "Yes" or "No" indicating whether the response is related to the specified role, 
               and the second element is the complete text generated by the text generator.
    Notes:
        - The instruction for the generator explicitly asks to determine if the response text is related to the role defined in `cfg`.
        - The evaluation criteria include checking the relationship to the role as well as ensuring the response is not overly similar to the baseline.
        - In case of any exceptions during the text generation process, the function catches the exception and returns (None, "").
    """
    instruction = f"""Quick {cfg.role} Check

Text to evaluate: '{response}'

Baseline reference:
'{baseline}'

Does the response contain {cfg.role}-related content? It doesn't metter the quality of the response, just if it is related to {cfg.role} or not.
The text might be repetitive or not make sense, just assess if it is related to the wide {cfg.role}-concept or domain.
Think step by step and answer the question below.
Please also consider the baseline reference if the phrase is too similar to baseline (just citing UNICODE, or just citing minus sign) probably it is not related to {cfg.role}.
Reply final answer format: [Yes] or [No]
"""
    try:
        completion = await generator.generate(instruction)
        if "[Yes]" in completion:
            return "Yes", completion
        elif "[No]" in completion:
            return "No", completion
    except Exception as e:
        return None, ""
