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
@click.option("--as-json", is_flag=True, help="Print as JSON")
@click.pass_context
def ls(ctx: click.Context, *, as_json: bool):
    """List registered models."""
    registry: ModelRegistry = ctx.obj["registry"]
    if not as_json:
        click.echo("\n".join(str(m) for m in registry.models))
    else:
        for m in registry.models:
            click.echo(json.dumps(dataclasses.asdict(m)))


@cli.command()
@click.option(
    "--model-id",
    required=True,
    help="Number of the model to get. See `ls` to list model numbers",
    type=int,
)
@click.pass_context
def get(ctx: click.Context, *, model_id: int):
    """Retrieve the model and configuration.

    Outputs JSON with model configuration, path to the model, and origin of the model.
    This downloads the pretrained model if necessary.
    """
    registry: ModelRegistry = ctx.obj["registry"]
    if model_id not in registry.model_ids:
        raise click.ClickException(
            f"'{model_id}' not found, available models are {registry.model_ids}. Use `wsinfer_zoo ls` to list all models."
        )

    registered_model = registry.get_model_by_id(model_id)
    model = registered_model.load_model()
    model_dict = dataclasses.asdict(model)
    model_json = json.dumps(model_dict)
    click.echo(model_json)
