# Git Feature Branch Workflow

Git tracks changes to files as a series of commits. A feature branch workflow keeps the main branch stable by doing new work on separate branches.

## Basic cycle
Create a branch with git switch -c feature/login-page. Make changes, stage them with git add, and record them with git commit -m "message". Push the branch with git push -u origin feature/login-page and open a pull request for review.

## Writing good commits
A good commit is small and focused on one change. The first line of the message summarises the change in about fifty characters, written in the imperative mood, for example "Add login form validation". A blank line and a longer explanation can follow when the reason for the change is not obvious.

## Merging versus rebasing
git merge joins two branches and keeps the full history, including a merge commit. git rebase replays your commits on top of another branch, producing a straight line of history. Never rebase commits that teammates have already pulled, because it rewrites history.

## Resolving conflicts
A conflict happens when two branches change the same lines. Git marks the file with conflict markers. Edit the file to keep the correct content, remove the markers, stage the file and finish the merge or rebase.

## Undoing changes
git restore discards uncommitted edits to a file. git revert creates a new commit that reverses an earlier commit, which is safe for shared branches. git reset moves the branch pointer and can discard commits, so use it only on local work.
