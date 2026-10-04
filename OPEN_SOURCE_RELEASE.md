# Preparing the public source release

This checkout combines source with private local work, training data, downloaded
artifacts, and historic product code. It must not be published by changing the
existing private repository's visibility. Ignoring or deleting a tracked file
does not remove it from old commits.

## Distribution boundary

The public snapshot includes software, public documentation, configuration
examples, synthetic test fixtures, and required component licenses. It excludes:

- `.work/`, `artifacts/`, `local/`, environments, caches, checkpoints, outputs,
  Rust build products, session/database state, and credentials.
- `data/` and the root training corpora, including imported paper/repository
  text and derived datasets whose redistribution rights were not established.
- Undocumented font binaries, the NVIDIA installer, imported `library/`
  fragments, and upstream model templates/recipe pending license review.
- Historic `8bitlabs.ai/` product code and the original Git history.

Local excluded files are preserved on disk. Only release exposure and tracking
are changed. Public font aliases use installed system fonts. Notebook outputs
and widget caches are cleared. `/path/to` values in examples must be replaced
with paths appropriate to your own installation. Excluded model templates
must be obtained from their upstream repository and reviewed for the intended
use before running the affected training recipes.

The policy is in `configs/public_release_policy.json`. `.gitignore` and
`.gitattributes` provide additional safeguards, but neither replaces the audit.

## Build and check a candidate

Install `gitleaks` (the initial audit used version 8.30.1). No model downloads,
GPU jobs, wallet signing, or network publication are needed for this process.

```bash
python3 -m unittest discover -s tests -p test_public_release.py
python3 scripts/prepare_public_release.py --check \
  --output local/public-release-reviewed
```

The output path must be new. The tool selects indexed paths, checks the current
file bytes, rejects symlinks and private-state paths, requires bundled licenses,
rejects notebook execution state and personal paths, and runs default provider
secret rules with recursive decoding. The sole exact secret-value exception is
the public CLAWD token mint. Findings are recorded privately under `local/`;
secrets are never printed. No output directory is created when a gate fails.
Untracked additions must be reviewed and added to the index before export.

The result contains no `.git`, ignored local files, or audit reports. It can be
reviewed and used to initialize a separate public repository with a new initial
history. Keep the original repository private. Do not copy `.git` or attach the
original commits, branches, or tags to the public repository.

To audit reachable local Git history separately:

```bash
python3 scripts/prepare_public_release.py --check --history
```

The original checkout is expected to fail that historical secret gate because
excluded corpora contain unresolved credential-shaped samples. More importantly,
it also contains historic proprietary product paths. A zero secret scan alone
would not clear those intellectual-property boundaries. The CI workflow scans
the complete checked-out history of the repository where it runs; use it in
the clean public repository too. Ensure CI checks run before release and enable
required status checks through repository settings when publishing.

## Licensing and ownership

The current root license is MIT. It permits commercial reuse with its notice
requirements. Copyright remains with its holders, but this license is not a
promise that nobody can copy the code. A reciprocal license such as AGPL-3.0
is a possible future choice for code the maintainers have authority to license;
it does not erase earlier MIT permissions or replace upstream terms. See
[LICENSING.md](LICENSING.md), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Review ownership for every newly imported component, data source, and model
artifact. A repository license, model-card label, or citation cannot grant
rights the uploader does not own. Keep valuable private product implementations
and private datasets outside the published distribution.

## Evidence and limits

The initial preparation verified a clean source candidate, release boundary
tests, affected training/privacy regression tests, and the Model Kit static
build. The original GitHub repository was verified private during preparation.
The local reachable-history scan found 420 matches for triage; matches include
public identifiers, placeholders, and excluded corpus samples. They are not
420 confirmed live credentials. Redacted history reports remain local.

Pattern scanning does not prove ownership or absence of every possible secret.
Binary reference illustrations are not OCR-audited. Existing Hugging Face
uploads, deployed sites, container layers, forks, and remote release assets
are separate surfaces and are not certified by this source export. If an
actual credential is identified in historic material, revoke it at its provider;
do not rely on deletion or history rewriting. See [SECURITY.md](SECURITY.md).
