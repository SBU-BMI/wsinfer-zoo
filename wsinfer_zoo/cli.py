"""Command-line interface for the WSInfer model zoo."""

import dataclasses
import json
import sys
from pathlib import Path

import click
import huggingface_hub
import requests
import tabulate

from wsinfer_zoo.client import (
    WSINFER_ZOO_REGISTRY_DEFAULT_PATH,
    HF_CONFIG_NAME,
    HF_WEIGHTS_SAFETENSORS_NAME,
    HF_WEIGHTS_PICKLE_NAME,
    HF_TORCHSCRIPT_NAME,
    InvalidModelConfiguration,
    InvalidRegistryConfiguration,
    ModelRegistry,
    validate_model_zoo_json,
    validate_config_json,
)


@click.group()
@click.option(
    "--registry-file",
    type=click.Path(path_type=Path),
    help="Path to the JSON file listing the models in the WSInfer zoo.",
    default=WSINFER_ZOO_REGISTRY_DEFAULT_PATH,
    envvar="WSINFER_ZOO_REGISTRY",
)
@click.pass_context
def cli(ctx: click.Context, *, registry_file: Path):
    registry_file = registry_file.expanduser()
    if not registry_file.exists():
        raise click.ClickException(f"registry file not found: {registry_file}")
    with open(registry_file) as f:
        d = json.load(f)
    # Raise an error if validation fails.
    try:
        validate_model_zoo_json(d)
    except InvalidRegistryConfiguration as e:
        raise InvalidRegistryConfiguration(
            "Registry schema is invalid. Please contact the developer by"
            " creating a new issue on our GitHub page:"
            " https://github.com/SBU-BMI/wsinfer-zoo/issues/new."
        ) from e
    registry = ModelRegistry.from_dict(d)
    ctx.ensure_object(dict)
    ctx.obj["registry"] = registry


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Print as JSON lines")
@click.pass_context
def ls(ctx: click.Context, *, as_json: bool):
    """List registered models.

    If not a TTY, only model names are printed. If a TTY, a pretty table
    of models is printed.
    """
    registry: ModelRegistry = ctx.obj["registry"]
    if as_json:
        for m in registry.models.values():
            click.echo(json.dumps(dataclasses.asdict(m)))
    else:
        if sys.stdout.isatty():
            info = [
                [m.name, m.description, m.hf_repo_id, m.hf_revision]
                for m in registry.models.values()
            ]
            click.echo(
                tabulate.tabulate(
                    info,
                    headers=["Name", "Description", "HF Repo ID", "Rev"],
                    tablefmt="grid",
                    maxcolwidths=[None, 24, 30, None],
                )
            )
        else:
            # You're being piped or redirected
            click.echo("\n".join(str(m) for m in registry.models))


@cli.command()
@click.option(
    "--model-name",
    required=True,
    help="Number of the model to get. See `ls` to list model names",
)
@click.pass_context
def get(ctx: click.Context, *, model_name: str):
    """Retrieve a model and its configuration.

    Outputs JSON with model configuration, path to the model, and origin of the model.
    The pretrained model is downloaded to a cache and reused if it is already present.
    """
    registry: ModelRegistry = ctx.obj["registry"]
    if model_name not in registry.models.keys():
        raise click.ClickException(
            f"'{model_name}' not found, available models are"
            " {list(registry.models.keys())}. Use `wsinfer_zoo ls` to list all"
            " models."
        )

    registered_model = registry.get_model_by_name(model_name)
    model = registered_model.load_model_torchscript()
    model_dict = dataclasses.asdict(model)
    model_json = json.dumps(model_dict)
    click.echo(model_json)


@cli.command()
@click.argument("input", type=click.File("r"))
def validate_config(*, input):
    """Validate a model configuration file against the JSON schema.

    INPUT is the config file to validate.

    Use a dash - to read standard input.
    """
    try:
        c = json.load(input)
    except Exception as e:
        raise click.ClickException(f"Unable to read JSON file. Original error: {e}")

    # Raise an error if the schema is not valid.
    try:
        validate_config_json(c)
    except InvalidRegistryConfiguration as e:
        raise InvalidModelConfiguration(
            "The configuration is invalid. Please see the traceback above for details."
        ) from e
    click.secho("Configuration file is VALID", fg="green")


@cli.command()
@click.argument("huggingface_repo_id")
@click.option("-r", "--revision", help="Revision to validate", default="main")
def validate_repo(*, huggingface_repo_id: str, revision: str):
    """Validate a repository on HuggingFace.

    This checks that the repository contains all of the necessary files and that
    the configuration JSON file is valid.
    """
    repo_id = huggingface_repo_id
    del huggingface_repo_id

    try:
        files_in_repo = list(
            huggingface_hub.list_files_info(repo_id=repo_id, revision=revision)
        )
    except huggingface_hub.utils.RepositoryNotFoundError:
        click.secho(
            f"Error: huggingface_repo_id '{repo_id}' not found on the HuggingFace Hub",
            fg="red",
        )
        sys.exit(1)
    except huggingface_hub.utils.RevisionNotFoundError:
        click.secho(
            f"Error: revision {revision} not found for repository {repo_id}",
            fg="red",
        )
        sys.exit(1)
    except requests.RequestException as e:
        click.echo("Error with request: {e}")
        click.echo("Please try again.")
        sys.exit(2)

    file_info = {f.rfilename: f for f in files_in_repo}

    repo_url = f"https://huggingface.co/{repo_id}/tree/{revision}"

    filenames_and_help = [
        (
            HF_CONFIG_NAME,
            "This file is a JSON file with the configuration of the model and includes"
            " necessary information for how to apply this model to new data. You can"
            " validate this file with the command 'wsinfer_zoo validate-config'.",
        ),
        (
            HF_TORCHSCRIPT_NAME,
            "This file is a TorchScript representation of the model and can be made"
            " with 'torch.jit.script(model)' followed by 'torch.jit.save'. This file"
            " contains the pre-trained weights as well as a graph of the model."
            " Importantly, it does not require a Python runtime to be used."
            f" Then, upload the file to the HuggingFace model repo at {repo_url}",
        ),
        (
            HF_WEIGHTS_PICKLE_NAME,
            "This file contains the weights of the pre-trained model in normal PyTorch"
            " format. Once you have a trained model, create this file with"
            f'\n\n    torch.save(model.state_dict(), "{HF_WEIGHTS_PICKLE_NAME}")'
            f"\n\n    Then, upload the file to the HuggingFace model repo at {repo_url}",
        ),
        (
            HF_WEIGHTS_SAFETENSORS_NAME,
            "This file contains the weights of the pre-trained model in SafeTensors"
            " format. The advantage of this file is that it does not have security"
            " concerns that Pickle files (pytorch default) have. To create the file:"
            "\n\n    from safetensors.torch import save_file"
            f'\n     save_file(model.state_dict(), "{HF_WEIGHTS_SAFETENSORS_NAME}")'
            f"\n\n    Then, upload the file to the HuggingFace model repo at {repo_url}",
        ),
    ]

    invalid = False
    for name, help_msg in filenames_and_help:
        if name not in file_info:
            click.secho(
                f"Required file '{name}' not found in HuggingFace model repo '{repo_id}'",
                fg="red",
            )
            click.echo(f"    {help_msg}")
            click.echo("-" * 40)
            invalid = True

    if invalid:
        click.secho(
            f"Model repository {repo_id} is invalid. See above for details.", fg="red"
        )
        sys.exit(1)

    config_path = huggingface_hub.hf_hub_download(
        repo_id, HF_CONFIG_NAME, revision=revision
    )
    with open(config_path) as f:
        config_dict = json.load(f)
    try:
        validate_config_json(config_dict)
    except InvalidModelConfiguration as e:
        click.secho(
            "Model configuration JSON file is invalid. Use 'wsinfer_zoo validate-config'"
            " with the configuration file to debug this further.",
            fg="red",
        )
        click.secho(
            f"Model repository {repo_id} is invalid. See above for details.", fg="red"
        )
        sys.exit(1)

    click.secho(f"Repository {repo_id} is VALID.", fg="green")
