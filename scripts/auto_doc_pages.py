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
with mkdocs_gen_files.open("reference/index.md", "a+") as file:
    file.write("::: moll")

# Create pages for public API objects
for module_name, module_obj in getmembers(moll, ismodule):
    if module_name.startswith("_"):
        continue

    # Create module index page
    with mkdocs_gen_files.open(
        module_index_path := f"reference/{module_name}/index.md", "a+"
    ) as file:
        print(f"::: moll.{module_name}", file=file)

    for public_name, _public_obj in getmembers(module_obj):
        if public_name.startswith("_"):
            continue

        # Create page for public object
        with mkdocs_gen_files.open(
            ref_path := f"reference/{module_name}/{public_name}.md", "a+"
        ) as file:
            file.write(f"::: moll.{module_name}.{public_name}\n")

        nav[("API", module_repr(module_name), attr_repr(public_name))] = ref_path

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
