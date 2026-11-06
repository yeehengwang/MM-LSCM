#!/usr/bin/env bash
# One-shot: init git (if needed), commit, create GitHub repo MM-LSCM, push, enable Pages from /docs
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) not found. Install: brew install gh && gh auth login"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Not logged in. Run: gh auth login"
  exit 1
fi

USER=$(gh api user --jq .login)
echo "GitHub user: $USER"

if [ ! -d .git ]; then
  git init -b main
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "$(cat <<'EOF'
Initial release of MM-LSCM code, demo data, README, and project page.
EOF
)"
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote origin already set: $(git remote get-url origin)"
else
  gh repo create MM-LSCM --public --source=. --remote=origin \
    --description "MM-LSCM: Multi-Modal Neural Radio Radiance Field for Localized Statistical Channel Modelling (IEEE GC Wkshps 2025)"
fi

git push -u origin HEAD

# Enable GitHub Pages from /docs on main
gh api -X POST "repos/$USER/MM-LSCM/pages" \
  -f build_type=legacy \
  -f source[branch]=main \
  -f source[path]=/docs 2>/dev/null \
  || gh api -X PUT "repos/$USER/MM-LSCM/pages" \
       -f build_type=legacy \
       -f source[branch]=main \
       -f source[path]=/docs 2>/dev/null \
  || echo "Pages setup skipped (enable manually: Settings → Pages → Deploy from branch main /docs)."

echo
echo "Repo:  https://github.com/$USER/MM-LSCM"
echo "Page:  https://$USER.github.io/MM-LSCM/"
