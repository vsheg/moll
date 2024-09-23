"""
Command line interface for the package.
"""

import typer

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


cli = typer.Typer(
    context_settings=CONTEXT_SETTINGS,
)


@cli.command()
def test():
    pass
