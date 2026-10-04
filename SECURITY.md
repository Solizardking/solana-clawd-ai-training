# Security reporting and public releases

Report sensitive findings using the repository's private vulnerability reporting
feature if enabled. Otherwise contact a maintainer through an established
private channel. Do not put credentials, wallet keypairs, personal data, or
exploit details in a public issue. No private reporting endpoint is assumed
to exist merely because this file is present.

The model is the Brain; signing belongs to the Hands. Never include wallet
keypairs in prompts, datasets, source, fixtures, logs, screenshots, or commits.
Configuration reads credentials from environment variables or local secret
stores. Environment examples must contain placeholders only.

Before publication use `scripts/prepare_public_release.py`. A clean working
tree scan does not clear Git history, tags, branches, forks, caches, release
assets, Docker layers, Spaces, datasets, model repositories, or deployed static
sites. A clean history-free snapshot does not repair those surfaces either.

If a real credential was exposed, revoke or rotate it at the provider. Deleting
a file or rewriting history is not credential revocation. Never publish the
original `.git` directory with the prepared snapshot.
