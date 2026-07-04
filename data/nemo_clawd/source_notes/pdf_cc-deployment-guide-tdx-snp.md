# Unified Deployment Guide for Confidential Computing

- Source ID: `pdf:cc-deployment-guide-tdx-snp`
- Source type: `pdf`
- Primary theme: `confidential_execution`
- Themes: `confidential_execution, operator_learning`
- Pages: 36
- Chunks: 11
- SHA-256: `5d68050885b8c979d81f73779d673a7cbfa390f6f8aafa818dd1a1ece7b5ccf5`
- Source path: `/Users/8bit/drive/pdfs/cc-deployment-guide-tdx-snp.pdf`

## Aliases

- None

## Extracted Abstract Or Lead

Documentation History DU-12302-001_v7.1 Version Date Authors Description of Change 1.0 7/25/2023 Rob Nertney Initial Version for Early Access 2.0 8/30/2023 Rob Nertney Minor fixes. EA2 Updates for Kata/CoCo and TDX installs 3.0 2/22/2024 Rob Nertney GA Version Release 4.0 7/09/2024 Rob Nertney Updating instructions from MVP Intel stack to more upstreamable flows. 5.0 2/25/2025 Rob Nertney Multi GPU integration; updating Intel paths for patched solutions. 6.0 5/15/2025 Rob Nertney Unified AMD/Intel guides. Updated for upstream 25.04 hosts. 6.1 6/17/2025 Rob Nertney Minor Links Update 7.0 1/7/2026 Rob Nertney Updated for Blackwell-architecture GPUs 7.1 4/6/2026 Rob Nertney Updated nomenclature Table of Contents Using This Guide.......................................................................................................................................................4 Document Structure....................................................................................................................................... 5 Supported Combinations of Hardware and Software.................................................................... 5 Hardware IT Administrator.............

## Training Use

training secure executor, attestation, isolation, and verifiable-compute reasoning

## Guardrail

Do not treat TEEs or fraud proofs as complete safety; require threat modeling, attestation evidence, and last-mile execution controls.

## Evaluation Focus

Ask for concrete trust boundaries, deployment assumptions, and what still needs operator verification.

## Headings

- Deployment Guide for Confidential
- Computing
- Documentation History
- Version Date Authors Description of Change
- Table of Contents
- Using This Guide
- Hopper-architecture GPUs in either Confidential Compute (CC) or Protected PCIe (PPCIe)
- The following personas have been defined:
- Document Structure
- Supported Combinations of Hardware
- Hopper GPUs
- CPU Vendor Confidential Computing Mode Host OS Guest OS
- Confidential Computing
- Genoa (LTS)
- Protected PCIe
- Intel (Single GPU)
- Rapids Protected PCIe
- Blackwell GPUs
- RTX PRO 6000
- Single GPU
