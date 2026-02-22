# Skill: @git-pro
# Description: High-trust, autonomous Git lifecycle management with validation.

## When to use:
- Trigger this skill when the task involves pushing changes to a remote repository.
- Use this when completing a feature branch or a bug fix.

## Architectural Constraints:
1. **Pre-Commit Audit**: NEVER push code without running the internal `npm run build` or `python -m pytest` (whichever is relevant to the workspace).
2. **Conventional Commits**: All commit messages must follow the format: `<type>(<scope>): <subject>`.
   - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.

## Actionable Execution Steps:
1. **Analyze**: Identify all modified files and group them by logical change.
2. **Validation**: Execute the workspace-defined test suite. If it fails, STOP and report the error; do not commit.
3. **Staging**: Stage files logically (avoid `git add .` if unrelated changes exist).
4. **Documentation**: Generate a concise but technical summary for the commit description based on the code diff.
5. **Execution**: Perform the commit and push to the current upstream branch.

## Success Criteria:
- Clean build/test pass recorded in the terminal.