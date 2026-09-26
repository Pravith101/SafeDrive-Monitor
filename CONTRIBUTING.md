# Contributing with your own GitHub account

Each teammate should use their own machine or Windows account, clone the repository, and make a real, reviewable change. Do not share passwords or access tokens, edit commit authorship to credit someone else, or create empty commits to simulate participation.

## Configure Git

In the teammate's own repository clone, set the name and email attached to that person's GitHub account. Use an email verified by GitHub (or that account's GitHub-provided `noreply` address):

```powershell
git config --local user.name "Their GitHub display name"
git config --local user.email "Their verified GitHub email"
git config --show-origin --get user.name
git config --show-origin --get user.email
```

Authenticate that account through Git Credential Manager's browser sign-in. The teammate should run this themselves and confirm the displayed account:

```powershell
git credential-manager github login
git credential-manager github list
```

The Git author name/email identify a commit; the credential-manager account authenticates a push. Both must belong to the teammate making the contribution. Never paste credentials into chat or a terminal command.

## Make and publish a contribution

Agree on a specific task, implement or review it, then create a branch and commit only the files changed for that task:

```powershell
git switch -c team/short-task-name
git status --short
git add path\to\changed-file.py path\to\related-test.py
git commit -m "Describe the contribution"
git show --format=fuller --stat HEAD
git push -u origin team/short-task-name
```

Do not stage datasets, model weights, generated arrays, local environments, credentials, or editor state. Keep test and evaluation results tied to the exact data and split used. A commit should be attributed only to people who authored or explicitly approved its changes.
