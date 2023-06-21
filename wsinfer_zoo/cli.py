"""Command-line interface for the WSInfer model zoo."""

import dataclasses
import json
import sys
from pathlib import Path

import click
import jsonschema
import tabulate

from wsinfer_zoo.client import (
    MODEL_CONFIG_SCHEMA,
    WSINFER_ZOO_REGISTRY_DEFAULT_PATH,
    WSINFER_ZOO_SCHEMA,
    InvalidModelConfiguration,
    InvalidRegistryConfiguration,
    ModelRegistry,
)

_here = Path(__file__).parent.resolve()


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
        jsonschema.validate(instance=d, schema=WSINFER_ZOO_SCHEMA)
    except jsonschema.ValidationError as e:
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
@click.option(
    "--input",
    help="Config file to validate (default is standard input)",
    type=click.File("r"),
    default=sys.stdin,
)
def validate_config(*, input):
    """Validate a model configuration file against the JSON schema."""
    try:
        c = json.load(input)
    except Exception as e:
        raise click.ClickException(f"Unable to read JSON file. Original error: {e}")

    # Raise an error if the schema is not valid.
    try:
        jsonschema.validate(instance=c, schema=MODEL_CONFIG_SCHEMA)
    except jsonschema.ValidationError as e:
        raise InvalidModelConfiguration(
            "The configuration is invalid. Please see the traceback above for details."
        ) from e
    click.secho("Passed", fg="green")
