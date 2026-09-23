#!/bin/sh
# Fail if pages/ is stale: every published literate HTML must be copied
# into pages/literate/ unchanged, and every local link in pages/index.html
# must resolve.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
status=0
for html in docs/literate/*.html; do
    if ! cmp -s "$html" "pages/literate/$(basename "$html")"; then
        echo "check-pages: pages/literate/$(basename "$html") is stale (run scripts/build-pages.sh)" >&2
        status=1
    fi
done
for link in $(grep -o 'href="[^"#:]*"' pages/index.html | sed 's/href="//; s/"$//' | sort -u); do
    [ -e "pages/$link" ] || { echo "check-pages: broken link in pages/index.html: $link" >&2; status=1; }
done
[ -f pages/build-info.json ] || { echo "check-pages: pages/build-info.json missing" >&2; status=1; }
[ $status -eq 0 ] && echo "check-pages: pages/ is current"
exit $status
