@AGENTS.md

Claude Code specifics:
- Use a git worktree per task (see AGENTS.md rule 1); subagents may split a task but share its branch.
- Queue GPU jobs with `tsp`; don't run long training in the foreground shell.
