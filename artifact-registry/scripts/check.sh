#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.elan/bin:${PATH}"
(cd formal/lean && lake build && lake exe vectors > ../vectors/preimages.json)
node --input-type=module -e '
  import { readFileSync, writeFileSync } from "node:fs";
  const p = "formal/vectors/preimages.json";
  const raw = readFileSync(p, "utf8").trim();
  writeFileSync(p, JSON.stringify(JSON.parse(raw), null, 2) + "\n");
'
node --test tests/*.test.mjs
echo "artifact-registry checks passed"
