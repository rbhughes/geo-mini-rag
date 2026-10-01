#!/usr/bin/env bash
# Deploy the site to Cloudflare Pages (rag.purr.io).
#
# site/ is hand-written static files, not a build -- there is nothing to
# compile, so the directory ships as it stands.
#
# --branch=main is load-bearing: Pages treats any other branch as a preview
# deployment, which gets its own *.pages.dev URL and leaves rag.purr.io
# pointing at whatever was last pushed to main.
set -euo pipefail
cd "$(dirname "$0")/.."

npx wrangler pages deploy site \
  --project-name geo-mini-rag \
  --branch=main

echo "Deployed. https://rag.purr.io"
