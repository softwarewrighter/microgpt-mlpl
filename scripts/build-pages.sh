#!/bin/sh
# Build the static site in pages/ LOCALLY (the GitHub workflow only
# uploads what is committed): copy the published literate HTML and stamp
# build-info.json. Run `just literate` first if the .org files changed.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
mkdir -p pages/literate
for html in docs/literate/*.html; do
    cp "$html" pages/literate/
done
commit=$(git rev-parse --short HEAD)
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"commit": "%s", "built_at": "%s"}\n' "$commit" "$built_at" > pages/build-info.json
touch pages/.nojekyll
echo "pages/ built ($commit): $(ls pages/literate | tr '\n' ' ')"
