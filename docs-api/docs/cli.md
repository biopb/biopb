# `biopb` CLI

The `biopb` console script, generated from its actual Typer command tree --
including the `algorithm` group (the control's registry) and the `tensor` and `image` subcommand groups
(`biopb.tensor.cli`/`biopb.image.cli`), which are lazily-imported sub-apps of
`biopb.cli:app` and so render here too.

::: mkdocs-typer2
    :module: biopb.cli
    :name: biopb
