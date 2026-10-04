# Third-party source and artifact boundaries

This inventory covers bundled source identified in the release audit. It is
not a declaration that all downloaded dependencies or training sources share
one license. Installed dependencies remain subject to their own terms.

| Component | Applicable license evidence | Attribution / boundary |
| --- | --- | --- |
| `trading_factory/cufolio/` | Its `LICENSE` (Apache-2.0), source SPDX headers, `LICENSE-3rd-party.txt` | NVIDIA CORPORATION & AFFILIATES; preserve bundled dependency notices. |
| `trading_factory/clawd-autoresearch-wiki/nanochat-master/` | Its `LICENSE` (MIT) | Copyright (c) 2025 Andrej Karpathy. |
| `trading_factory/clawd-autoresearch-wiki/solana-chat/` | Its `LICENSE` (MIT) | nanochat-derived Solana adaptation; upstream notice remains applicable. |
| Autoresearch-derived `prepare.py`, `train.py`, `program.md`, `analysis.ipynb`, and project/dependency metadata in `trading_factory/clawd-autoresearch-wiki/` | [Upstream README license statement](https://github.com/karpathy/autoresearch#license): MIT; full text bundled in `LICENSES/Autoresearch-MIT.txt` | Andrej Karpathy and contributors. Attribution restored for imported experiment scaffolding; Clawd integrations remain subject to their applicable notices. |
| NVIDIA-marked transaction-foundation files | Individual `SPDX-License-Identifier: Apache-2.0`; full text in `LICENSES/Apache-2.0.txt` | NVIDIA CORPORATION & AFFILIATES; Clawd adaptations do not remove upstream rights. |
| `space/` demonstration source | Existing Space README metadata specifies Apache-2.0; full text in `space/LICENSE` | Preserve this component's existing licensing choice; referenced model weights retain their own terms. |
| Model cards and external model references | Referenced model repository terms | No model weights or adapters are bundled in the public source snapshot. Resolve the exact model revision and license before redistribution. |
| Dataset cards and download/ingestion integrations | Artifact-specific provenance and licenses | Generated corpora and imported research text are excluded. Source availability and citations do not establish redistribution rights. |
| Solana and other third-party logos / names | Third-party marks; no endorsement implied | Root MIT software licensing does not authorize adopting another party's branding. |
| Bundled `.otf` fonts | No accompanying font redistribution license found | Excluded from the index and public snapshot; frontend font aliases use installed system fonts. |
| `ngccli_mac_intel.pkg` | NVIDIA installer; distribution rights not established here | Excluded. Obtain software from the provider. |
| Imported model templates / upstream training recipe | Model-specific redistribution terms not resolved | `configs/nemotron35_chat_template_genmask.jinja`, `deploy/nemotron/training/assistant-mask.jinja`, and `upstream-lora.yaml` excluded; obtain/review from the upstream model repository. |
| `library/` imported tooling fragments | Provenance and redistribution terms not established | Excluded pending review. |

Public release preparation clears notebook execution output and widget state.
Original source cells and license headers remain. Font aliases were changed to
use system-installed fonts. Personal workstation paths in examples were replaced
with `/path/to` placeholders. These are release-preparation modifications, not
claims of ownership over upstream source.

Do not replace an upstream license, claim exclusive ownership of upstream code,
or use the root license to redistribute material excluded here. New imports
must add their provenance, exact revision, license, and any required notices.
