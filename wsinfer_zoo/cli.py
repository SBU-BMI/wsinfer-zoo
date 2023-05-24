"""Command-line interface for the WSInfer model zoo."""

import dataclasses
import json

import click

from wsinfer_zoo.client import list_model_names, load_model_from_name


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--model-name",
    type=click.Choice(list_model_names()),
    required=True,
    help="Name of model to get.",
)
def get(*, model_name: str):
    """Retrieve the model and configuration.

    Outputs JSON with model configuration, path to the model, and origin of the model.
    """
    frozenmodel = load_model_from_name(model_name)
    if frozenmodel.model_path is None:
        raise click.ClickException(
            "Path to the model is unknown. Please contact the developer by creating a new issue on our GitHub repository."
        )
    frozenmodel_dict = dataclasses.asdict(frozenmodel)
    frozenmodel_json_str = json.dumps(frozenmodel_dict)
    click.echo(frozenmodel_json_str)
