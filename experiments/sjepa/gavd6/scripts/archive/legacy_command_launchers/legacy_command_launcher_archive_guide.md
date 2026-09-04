# Archived command launchers

These files preserve pre-CLI invocation paths for completed experiments. They
contain no scientific implementation and should not be used in new Slurm jobs,
notebooks, or documentation.

Use the corresponding `uv run gavd6 ...` command from the project root. Run
`uv run gavd6 --help` to find the maintained command. The launchers remain here
only so an old runbook can be translated or reproduced deliberately.

Archive policy:

- fixes that preserve reproducibility are allowed;
- new features and research logic are not;
- active code must never import these files;
- a launcher may be deleted once no retained artifact or historical runbook
  names it.
