# Role-Vectors

## Overview

Role-Vectors is a Python-based pipeline for identifying and extracting "role vectors" from large language models. These vectors represent specific roles or personas (e.g., "mathematician," "doctor," "lawyer") and can be used to steer model behavior, enabling more controlled and context-aware text generation.

The project implements a methodology to generate direction vectors by contrasting the model's internal representations of role-specific text with a general-purpose baseline. The most effective direction is then selected and can be applied as an intervention during inference.

## Features

*   **Direction Vector Generation:** Creates candidate vectors that encapsulate the essence of a specific role.
*   **Automated Vector Selection:** Evaluates and selects the optimal direction vector based on performance on a validation set.
*   **Model Intervention:** Applies the selected vector to steer model outputs during generation.
*   **Extensible Model Support:** Easily adaptable to new models through a factory pattern (currently supports Gemma, Llama, Qwen, and Yi).
*   **HPC-Ready:** Includes example scripts for running the pipeline on SLURM-based clusters.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Crisp-Unimib/Role-Vectors.git
    cd Role-Vectors
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r target_direction/requirements.txt
    ```

4.  **Set up environment variables:**
    The pipeline uses an external API for testing the generated vectors. Create a `.env` file in the `target_direction` directory by copying the example:
    ```bash
    cp target_direction/.env.example target_direction/.env
    ```
    Then, add your OpenRouter API key to the `.env` file:
    ```
    OPENROUTER_KEY="your-api-key"
    ```

## Dataset

The pipeline relies on two types of datasets:

*   **Base Dataset:** A general-purpose instruction dataset used as a baseline. The default is `alpaca.json`.
*   **Target Dataset:** Role-specific datasets containing instructions tailored to a particular persona (e.g., a mathematician, a doctor).

All datasets are located in `target_direction/pipeline/dataset/processed` and are expected to be in JSON format, with each entry containing an `"instruction"` field.

### Adding New Roles

To add a new role, you need to create corresponding train and test splits in the `target_direction/pipeline/dataset/splits` directory, following the naming convention:

*   `target_train_<role_name>.json`
*   `target_test_<role_name>.json`

For example, for the role "artist," you would create `target_train_artist.json` and `target_test_artist.json`.

## Usage

The main entry point for the pipeline is `target_direction/pipeline/run_pipeline.py`. You can run it from the root of the project.

### Running the Pipeline for a Single Role

To generate a role vector for a single role, use the following command:

```bash
python -m target_direction.pipeline.run_pipeline --model_path <model_name_or_path> --role <role_name> --pos <position>
```

*   `--model_path`: The path or Hugging Face identifier for the model (e.g., `meta-llama/Llama-3.2-1B-Instruct`).
*   `--role`: The name of the role to process (e.g., `mathematician`).
*   `--pos`: The position for the intervention.

Example:
```bash
python -m target_direction.pipeline.run_pipeline --model_path "meta-llama/Llama-3.2-1B-Instruct" --role "mathematician" --pos -7
```

### Running for Multiple Roles

The `target_direction/pipeline/run_roles.sh` script provides an example of how to run the pipeline for a list of predefined roles:

```bash
bash target_direction/pipeline/run_roles.sh
```

### High-Performance Computing (HPC)

For large-scale experiments, the `target_direction/run_example_leonardo_hpc.sh` script shows how to adapt the pipeline for a SLURM-based HPC environment. You may need to modify it to fit your specific cluster configuration.

## Configuration

The pipeline's behavior can be customized through the `Config` class in `target_direction/pipeline/config.py`. Key parameters include:

*   `model_alias`: A short name for the model, used for creating artifact directories.
*   `model_path`: The path to the LLM.
*   `n_train`: Number of samples for training.
*   `n_test`: Number of samples for testing.
*   `max_new_tokens`: The maximum number of tokens to generate.
*   `role`: The role being processed.
*   `test`: The test set to use for validation.
*   `coeff`: The coefficient for the intervention.

## Supported Models

The project is designed to be model-agnostic. The following model architectures are currently supported:

*   Gemma
*   Gemma2
*   Llama2
*   Llama3
*   Qwen
*   Yi

You can add support for new models by creating a new model class in `target_direction/pipeline/model_utils` and updating the `model_factory.py`.

## Acknowledgements

The methodology and parts of the codebase are inspired by the work on refusal directions by Andy Arditi. We extend our gratitude for the foundational research and open-source contributions.

*   **Original Repository:** [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.