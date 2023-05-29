"""Command-line interface for the WSInfer model zoo."""

import dataclasses
import json
from pathlib import Path

import click

from wsinfer_zoo.client import WSINFER_ZOO_REGISTRY_DEFAULT_PATH, ModelRegistry


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
    registry = ModelRegistry.from_dict(d)
    ctx.ensure_object(dict)
    ctx.obj["registry"] = registry


@cli.command()
@click.pass_context
def ls(ctx: click.Context):
    """List registered models."""
    registry: ModelRegistry = ctx.obj["registry"]
    names = registry.model_names
    click.echo("\n".join(names))


@cli.command()
@click.option(
    "--model-name",
    required=True,
    help="Name of model to get.",
)
@click.pass_context
def get(ctx: click.Context, *, model_name: str):
    """Retrieve the model and configuration.

    Outputs JSON with model configuration, path to the model, and origin of the model.
    This downloads the pretrained model if necessary.
    """
    registry: ModelRegistry = ctx.obj["registry"]
    if model_name not in registry.model_names:
        raise click.ClickException(
            f"'{model_name}' not found, available model names are {registry.model_names}"
        )
    model = registry.load_model_from_name(model_name)
    model_dict = dataclasses.asdict(model)
    model_json = json.dumps(model_dict)
    click.echo(model_json)
