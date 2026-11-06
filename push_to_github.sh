#!/usr/bin/env bash
# Upload MM-LSCM to your GitHub (create public repo + push + Pages from /docs)
set -euo pipefail
cd "$(dirname "$0")"

pick_gh() {
  for c in gh /tmp/gh_2.67.0_macOS_arm64/bin/gh /opt/homebrew/bin/gh /usr/local/bin/gh; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

if ! GH_BIN=$(pick_gh); then
  echo "GitHub CLI not found."
  echo "Install:  brew install gh"
  echo "Or download: https://cli.github.com/"
  echo "A portable binary may already be at /tmp/gh_2.67.0_macOS_arm64/bin/gh after curl install."
  exit 1
fi

echo "Using: $GH_BIN"

if ! "$GH_BIN" auth status >/dev/null 2>&1; then
  echo "Logging into GitHub (browser)..."
  "$GH_BIN" auth login --hostname github.com --git-protocol https --web
fi

USER=$("$GH_BIN" api user --jq .login)
echo "GitHub user: $USER"

if [ ! -d .git ]; then
  git init -b main
fi

git add -A
if ! git diff --cached --quiet || ! git rev-parse HEAD >/dev/null 2>&1; then
  if ! git rev-parse HEAD >/dev/null 2>&1; then
    git commit -m "Initial release of MM-LSCM: code, demo data, README, and project page."
  elif ! git diff --cached --quiet; then
    git commit -m "Update MM-LSCM release package."
  fi
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote origin: $(git remote get-url origin)"
else
  if "$GH_BIN" repo view "$USER/MM-LSCM" >/dev/null 2>&1; then
    git remote add origin "https://github.com/$USER/MM-LSCM.git"
  else
    "$GH_BIN" repo create MM-LSCM --public --source=. --remote=origin \
      --description "MM-LSCM: Multi-Modal Neural Radio Radiance Field for Localized Statistical Channel Modelling (IEEE GC Wkshps 2025)"
  fi
fi

git push -u origin HEAD

"$GH_BIN" api -X POST "repos/$USER/MM-LSCM/pages" \
  -f build_type=legacy -f source[branch]=main -f source[path]=/docs 2>/dev/null \
  || "$GH_BIN" api -X PUT "repos/$USER/MM-LSCM/pages" \
       -f build_type=legacy -f source[branch]=main -f source[path]=/docs 2>/dev/null \
  || echo "Enable Pages manually: Settings → Pages → Deploy from branch main /docs"

echo
echo "Repo: https://github.com/$USER/MM-LSCM"
echo "Page: https://$USER.github.io/MM-LSCM/"
