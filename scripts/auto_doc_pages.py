"""Generate the code reference pages and navigation."""

from inspect import getmembers, ismodule

import mkdocs_gen_files

import moll

nav = mkdocs_gen_files.Nav()  # type: ignore


def module_repr(name: str) -> str:
    return f"<code>.{name}</code>"


def attr_repr(name: str) -> str:
    return f"<code>{name}()</code>"


# Create package index page
with mkdocs_gen_files.open("api/index.md", "a+") as file:
    file.write("::: moll\n")

# Create pages for public API objects
for module_name, module_obj in getmembers(moll, ismodule):
    if module_name.startswith("_"):
        continue

    for public_name, _public_obj in getmembers(module_obj):
        if public_name.startswith("_"):
            continue

        # Create page for public object
        with mkdocs_gen_files.open(
            ref_path := f"api/{module_name}.md", "a+"
        ) as file:
            file.write(f"::: moll.{module_name}.{public_name}\n")

