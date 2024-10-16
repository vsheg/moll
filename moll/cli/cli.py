"""
Command line interface for the package.
"""

import typer

from .pick import cli_pick

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


cli = typer.Typer(
    context_settings=CONTEXT_SETTINGS,
)

cli.add_typer(cli_pick, name="pick")
