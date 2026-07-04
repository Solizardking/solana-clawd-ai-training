# Bitvm

- Source ID: `pdf:bitvm`
- Source type: `pdf`
- Primary theme: `confidential_execution`
- Themes: `confidential_execution`
- Pages: 8
- Chunks: 3
- SHA-256: `54afcbd994811aa88485fad8a9dc4397386924f9b32dd0147ffb2989a23b0d88`
- Source path: `/Users/8bit/drive/pdfs/bitvm.pdf`

## Aliases

- None

## Extracted Abstract Or Lead

BitVM is a computing paradigm to express Turing-complete Bitcoin contracts. This requires no changes to the network’s consensus rules. Rather than executing computations on Bitcoin, they are merely verified, similarly to optimistic rollups. A prover makes a claim that a given function evaluates for some particular inputs to some specific output. If that claim is false, then the verifier can perform a succinct fraud proof and punish the prover. Using this mechanism, any computable function can be verified on Bitcoin. Committing to a large program in a Taproot address requires significant amounts of off-chain computation and communication, however the resulting on-chain footprint is minimal. As long as both parties collaborate, they can perform arbitrarily complex, stateful off-chain computation, without leaving any trace in the chain. On-chain execution is required only in case of a dispute.

## Training Use

training secure executor, attestation, isolation, and verifiable-compute reasoning

## Guardrail

Do not treat TEEs or fraud proofs as complete safety; require threat modeling, attestation evidence, and last-mile execution controls.

## Evaluation Focus

Ask for concrete trust boundaries, deployment assumptions, and what still needs operator verification.

## Headings

- BitVM: Compute Anything on Bitcoin
- Robin Linus
- December 12, 2023
- Committing to a large program in a Taproot address requires significant amounts
- 1 Introduction
- 2 Architecture
- 3 Bit Value Commitment
- 4 Logic Gate Commitment
- 5 Binary Circuit Commitment
- 6 Challenges and Responses
- 7 Inputs and Outputs
- 8 Limitations and Outlook
- Other directions of research include cross-application memory, how to make statements
- 9 Conclusion
- Acknowledgments
- References
- Sponsor BitVM developers: bc1qf5g6z0py2t3t49gupeqrlewga0qz2etalu4xf9
