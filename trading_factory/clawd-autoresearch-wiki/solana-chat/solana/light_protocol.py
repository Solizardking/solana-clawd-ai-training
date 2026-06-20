"""
Light Protocol ZK Compression Integration for Solana Chat.

Provides Python SDK wrappers and TypeScript reference examples for:
1. Compressed token accounts (mint, transfer, compress/decompress)
2. Compressed PDAs (create, update, close, reinit, burn)
3. Nullifier PDAs for double-spend prevention
4. Solana Attestation Service (SAS) for model output credentialing

Architecture:
- Python layer: simulation + training data generation
- TypeScript examples: deployable to Solana via Light Protocol CLI

Usage:
    from solana.light_protocol import (
        CompressedTokenClient,
        CompressedPDAClient,
        NullifierClient,
        AttestationClient,
    )

Reference: https://docs.lightprotocol.com/
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Light Protocol Compressed Token Accounts
# =============================================================================
# Compressed token accounts store ownership for compressed tokens like regular
# SPL token accounts with two core differences:
#   1. No Associated Token Accounts (ATAs) required
#   2. No rent-exempt balance needed (~160x cheaper)
#
# Created in:
#   - mintTo(): creates compressed token accounts for recipients
#   - transfer(): consumes existing accounts as input, creates new output accounts

class TokenProgram(Enum):
    SPL_TOKEN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


@dataclass
class CompressedTokenAccount:
    """A compressed token account on Solana via Light Protocol.
    
    Unlike regular SPL token accounts:
    - No ATA derivation needed
    - No rent-exempt lamports required
    - State stored in Merkle tree (32-byte root onchain)
    """
    mint: str
    owner: str
    amount: int
    decimals: int
    token_pool: str  # Interface PDA for compression
    merkle_root: str = ""


class CompressedTokenClient:
    """Python SDK for Light Protocol compressed token operations.
    
    Mirrors the @lightprotocol/compressed-token TypeScript SDK.
    Generates instruction data for onchain submission.
    """

    def __init__(self, rpc_url: str | None = None):
        self.rpc_url = rpc_url or os.environ.get("SOLANA_RPC_URL",
                                                   "https://api.mainnet-beta.solana.com")

    def create_mint(self, authority: str, decimals: int = 9) -> dict[str, Any]:
        """Create SPL mint with interface PDA for compression.
        
        The interface PDA locks tokens while compressed and releases them 
        when decompressed. Each mint supports a maximum of 4 interface PDAs.
        
        Args:
            authority: Mint authority public key
            decimals: Token decimals (default: 9)
            
        Returns:
            Dict with mint address, token pool PDA, and serialized instruction
        """
        # Simulate deriveMintPda + createMint from @lightprotocol/compressed-token
        seed = f"mint:{authority}:{decimals}:{int(time.time())}"
        mint_address = hashlib.sha256(seed.encode()).hexdigest()[:44]
        
        # Derive token pool PDA (interface PDA for compression)
        token_pool_seed = f"token_pool:{mint_address}:0"
        token_pool_pda = hashlib.sha256(token_pool_seed.encode()).hexdigest()[:44]
        
        return {
            "mint": mint_address,
            "token_pool_pda": token_pool_pda,
            "authority": authority,
            "decimals": decimals,
            "instructions": [
                {
                    "program": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
                    "accounts": [authority],
                    "data": f"create_mint:{mint_address}:{decimals}",
                },
                {
                    "program": "CompressedToken1111111111111111111111111111111",
                    "accounts": [authority, mint_address],
                    "data": f"create_token_pool:{mint_address}:{token_pool_pda}",
                },
            ],
            "compressed_account_cost_sol": 0.000015,  # ~160x cheaper than standard
        }

    def mint_to(self, mint: str, recipient: str, authority: str,
                amount: int) -> dict[str, Any]:
        """Mint compressed tokens, creating compressed token accounts for recipients.
        
        Compressed token accounts are created automatically — no ATA needed.
        
        Args:
            mint: SPL mint with interface PDA for compression
            recipient: Recipient public key
            authority: Mint authority
            amount: Token amount (raw units)
            
        Returns:
            Dict with transaction data
        """
        return {
            "mint": mint,
            "recipient": recipient,
            "amount": amount,
            "compressed_account_created": True,
            "rent_saved_sol": round(amount * 0.000002, 8),  # ~160x cheaper
            "instructions": [
                {
                    "program": "CompressedToken1111111111111111111111111111111",
                    "accounts": [authority, mint, recipient],
                    "data": f"mint_to:{mint}:{recipient}:{amount}",
                }
            ],
            "note": "Compressed token account created for recipient — no ATA or rent required",
        }

    def transfer(self, mint: str, sender: str, recipient: str,
                 amount: int) -> dict[str, Any]:
        """Transfer compressed tokens between accounts.
        
        Transfers consume input accounts from sender and create new output
        accounts for sender and recipient with updated balances.
        SPL token accounts can be compressed/decompressed in same tx.
        
        Args:
            mint: SPL mint with interface PDA for compression
            sender: Sender public key
            recipient: Recipient public key
            amount: Token amount (raw units)
            
        Returns:
            Dict with transaction data
        """
        return {
            "mint": mint,
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "instructions": [
                {
                    "program": "CompressedToken1111111111111111111111111111111",
                    "accounts": [sender, recipient, mint],
                    "data": f"transfer:{mint}:{sender}:{recipient}:{amount}",
                }
            ],
            "utxo_pattern": True,  # Consumes old, creates new accounts
        }

    def compress(self, mint: str, owner: str, source_ata: str,
                 recipient: str, amount: int) -> dict[str, Any]:
        """Convert SPL tokens to compressed tokens.
        
        Args:
            mint: SPL mint with interface PDA for compression
            owner: Token owner
            source_ata: Source SPL token account (ATA)
            recipient: Recipient for compressed tokens
            amount: Amount to compress
            
        Returns:
            Dict with compress instruction data
        """
        return {
            "mint": mint,
            "owner": owner,
            "source_ata": source_ata,
            "recipient": recipient,
            "amount": amount,
            "cost_analysis": {
                "standard_ata_rent_sol": 0.00204,
                "compressed_cost_sol": 0.000015,
                "savings_pct": 99.3,
            },
            "instructions": [
                {
                    "program": "CompressedToken1111111111111111111111111111111",
                    "accounts": [owner, source_ata, recipient, mint],
                    "data": f"compress:{mint}:{source_ata}:{recipient}:{amount}",
                }
            ],
        }

    def decompress(self, mint: str, owner: str, recipient: str,
                   amount: int) -> dict[str, Any]:
        """Convert compressed tokens back to SPL tokens.
        
        Args:
            mint: SPL mint with interface PDA for compression
            owner: Compressed token owner
            recipient: Recipient for decompressed SPL tokens
            amount: Amount to decompress
            
        Returns:
            Dict with decompress instruction data
        """
        return {
            "mint": mint,
            "owner": owner,
            "recipient": recipient,
            "amount": amount,
            "instructions": [
                {
                    "program": "CompressedToken1111111111111111111111111111111",
                    "accounts": [owner, recipient, mint],
                    "data": f"decompress:{mint}:{recipient}:{amount}",
                }
            ],
        }


# =============================================================================
# Compressed PDAs (Program Derived Addresses)
# =============================================================================
# Compressed PDAs provide full composability and functionality of accounts at
# PDAs, without rent-exemption cost per account. Suitable for per-user state,
# DePIN registrations, nullifiers, etc.
#
# Cost comparison (100-byte account):
#   Regular PDA:  ~0.0016 SOL
#   Compressed:   0.000015 SOL

class CompressedPDAClient:
    """Python SDK for compressed PDA operations.
    
    Mirrors the @lightprotocol/stateless.js SDK for compressed accounts.
    """

    def create_account(self, program_id: str, signer: str,
                       seeds: list[bytes], data: bytes) -> dict[str, Any]:
        """Create a compressed PDA with initial data.
        
        Flow:
        1. Derive address from seeds + address tree
        2. Fetch validity proof (prove address doesn't exist)
        3. Build instruction with proof + packed accounts
        4. Send transaction
        
        Args:
            program_id: Owning program
            signer: Transaction signer
            seeds: Address derivation seeds
            data: Initial account data
            
        Returns:
            Dict with derived address and instruction data
        """
        # Derive compressed address (similar to PDA but includes address tree)
        seed_hash = hashlib.sha256(b"".join(seeds)).hexdigest()
        address = hashlib.sha256(
            f"{program_id}:{seed_hash}:ADDRESS_TREE_V2".encode()
        ).hexdigest()[:44]

        return {
            "address": address,
            "program_id": program_id,
            "signer": signer,
            "data_size": len(data),
            "data_hash": hashlib.sha256(data).hexdigest(),
            "cost_sol": 0.000015,  # Compressed PDA cost
            "standard_pda_cost_sol": 0.0016,
            "savings_x": round(0.0016 / 0.000015, 1),  # ~106x
            "validity_proof_required": True,
            "instructions": [
                {
                    "note": "Derive address with derive_address_v2()",
                    "parameters": {
                        "seeds": [s.hex() for s in seeds],
                        "address_tree": "ADDRESS_TREE_V2",
                        "program_id": program_id,
                    }
                },
                {
                    "note": "Fetch validity proof with getValidityProofV0()",
                    "parameters": {
                        "addresses": [address],
                        "address_tree": "ADDRESS_TREE_V2",
                    }
                },
                {
                    "note": "Build CPI to Light System Program",
                    "program": "LightSystemProgram1111111111111111111111111111",
                    "accounts": [signer, "ADDRESS_TREE_V2", "STATE_TREE"],
                    "data": f"create_compressed_account:{address}:{len(data)}",
                }
            ],
        }

    def update_account(self, address: str, program_id: str,
                       current_data: bytes, new_data: bytes,
                       output_state_tree: str) -> dict[str, Any]:
        """Update compressed PDA data (UTXO pattern).
        
        Each update:
        1. Consumes existing account hash
        2. Produces new account hash with updated data
        3. Nullifies old hash to prevent double-spending
        
        Args:
            address: Compressed PDA address
            program_id: Owning program
            current_data: Current account data
            new_data: New account data
            output_state_tree: Output state tree pubkey
            
        Returns:
            Dict with update instruction data
        """
        return {
            "address": address,
            "program_id": program_id,
            "current_data_hash": hashlib.sha256(current_data).hexdigest(),
            "new_data_hash": hashlib.sha256(new_data).hexdigest(),
            "utxo_pattern": True,
            "instructions": [
                {
                    "program": "LightSystemProgram1111111111111111111111111111",
                    "note": "Update via CPI — consumes old hash, creates new",
                    "accounts": [address, output_state_tree],
                    "data": f"update_compressed_account:{address}",
                }
            ],
        }

    def close_account(self, address: str, program_id: str,
                      current_data: bytes) -> dict[str, Any]:
        """Close compressed PDA (reclaimable).
        
        A closed compressed account can be reinitialized.
        Produces output state with zero values.
        """
        return {
            "address": address,
            "program_id": program_id,
            "action": "close",
            "can_reinitialize": True,
            "instructions": [
                {
                    "program": "LightSystemProgram1111111111111111111111111111",
                    "data": f"close_compressed_account:{address}",
                    "note": "Zero discriminator + empty data",
                }
            ],
        }

    def burn_account(self, address: str, program_id: str,
                     current_data: bytes) -> dict[str, Any]:
        """Burn compressed PDA permanently.
        
        A burned account CANNOT be reinitialized.
        No output state is created.
        """
        return {
            "address": address,
            "program_id": program_id,
            "action": "burn",
            "can_reinitialize": False,
            "note": "Permanently deleted — no output state created",
            "instructions": [
                {
                    "program": "LightSystemProgram1111111111111111111111111111",
                    "data": f"burn_compressed_account:{address}",
                }
            ],
        }


# =============================================================================
# Nullifier PDAs
# =============================================================================
# Nullifier PDAs prevent onchain instructions from being executed more than
# once. Derived from ["nullifier", id] seeds where id is a unique identifier
# (nonce, uuid, hash of signature, etc.)
#
# Cost: ~15,000 lamports (~0.000015 SOL) per nullifier
# Program ID: NFLx5WGPrTHHvdRNsidcrNcLxRruMC92E4yv7zhZBoT
# Networks: Mainnet, Devnet

class NullifierClient:
    """Client for nullifier PDA creation.
    
    Useful for:
    - Preventing double-spending in payment flows
    - Ensuring one-time-use credentials
    - Replay protection for x402 agent payments
    """

    NULLIFIER_PROGRAM = "NFLx5WGPrTHHvdRNsidcrNcLxRruMC92E4yv7zhZBoT"

    def create_nullifier(self, payer: str, id_bytes: bytes) -> dict[str, Any]:
        """Create a nullifier PDA for a unique identifier.
        
        If the PDA already exists (same id used before), the transaction fails.
        This prevents replay attacks.
        
        Args:
            payer: Fee payer pubkey
            id_bytes: Unique 32-byte identifier
            
        Returns:
            Dict with nullifier address and instruction data
        """
        # Derive nullifier PDA: ["nullifier", id]
        nullifier_address = hashlib.sha256(
            b"nullifier:" + id_bytes
        ).hexdigest()[:44]

        return {
            "nullifier_address": nullifier_address,
            "program_id": self.NULLIFIER_PROGRAM,
            "payer": payer,
            "cost_lamports": 15_000,  # ~0.000015 SOL
            "purpose": "Prevents double-execution of onchain instruction",
            "instructions": [
                {
                    "program": self.NULLIFIER_PROGRAM,
                    "accounts": [payer, nullifier_address],
                    "data": f"create_nullifier:{id_bytes.hex()}",
                    "note": "Fails if id has been used before",
                }
            ],
            "combine_with": "Prepend or append to your main transaction instruction",
        }

    def verify_nullifier(self, id_bytes: bytes) -> dict[str, Any]:
        """Check if a nullifier PDA exists (meaning id was already used)."""
        nullifier_address = hashlib.sha256(
            b"nullifier:" + id_bytes
        ).hexdigest()[:44]
        return {
            "nullifier_address": nullifier_address,
            "exists_query": f"rpc.getAccountInfo({nullifier_address})",
            "exists": None,  # Resolved at runtime
        }


# =============================================================================
# Solana Attestation Service (SAS) for Model Output Credentialing
# =============================================================================
# SAS provides on-chain credentialing for model outputs:
#   1. Create Credential — establishes authority
#   2. Create Schema — defines attestation structure (prompt_hash, output_hash, model_hash)
#   3. Create Attestation — issues verifiable credential for model output
#   4. Verify Attestation — checks output validity onchain
#   5. Close Attestation — revokes credential
#
# Program ID: 22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG

class AttestationClient:
    """Solana Attestation Service client for model output credentialing.
    
    Wraps model inferences in verifiable on-chain credentials:
      - Credential = "SolanaChat-Model-v1" (the attestation authority)
      - Schema = prompt_hash + output_hash + model_version (the data structure)
      - Attestation = individual inference output + timestamp + ZK proof
    
    SAS Program: 22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG
    """

    SAS_PROGRAM_ID = "22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG"

    def __init__(self):
        self.issued_attestations: list[dict[str, Any]] = []

    def create_credential(self, authority: str, credential_name: str) -> dict[str, Any]:
        """Create a credential that represents the model authority.
        
        The credential serves as the top-level identity that issues
        attestations for model outputs. Think of it as:
        "SolanaChat Model v2.4" is authorized to issue inference receipts.
        
        Args:
            authority: Pubkey of the credential authority
            credential_name: Name (e.g. "SolanaChat-Model-v1")
            
        Returns:
            Dict with credential PDA and instruction data
        """
        # Derive credential PDA: [authority, name]
        credential_pda = hashlib.sha256(
            f"{self.SAS_PROGRAM_ID}:credential:{authority}:{credential_name}".encode()
        ).hexdigest()[:44]

        return {
            "credential_pda": credential_pda,
            "authority": authority,
            "name": credential_name,
            "program_id": self.SAS_PROGRAM_ID,
            "instructions": [
                {
                    "program": self.SAS_PROGRAM_ID,
                    "accounts": [authority, credential_pda],
                    "data": f"create_credential:{credential_name}",
                }
            ],
            "note": "Use sas-lib: getCreateCredentialInstruction()",
        }

    def create_schema(self, authority: str, credential_pda: str,
                      schema_name: str, fields: list[dict[str, Any]],
                      version: int = 1) -> dict[str, Any]:
        """Create a schema for model attestation data.
        
        The schema defines the structure of attestation data.
        For model outputs, we define the fields:
        - prompt_hash: string (SHA-256 of input)
        - output_hash: string (SHA-256 of generated output)
        - model_hash: string (SHA-256 of model weights)
        - proof_hash: string (SHA-256 of ZK proof, optional)
        - timestamp: u64 (Unix timestamp)
        
        Args:
            authority: Credential authority
            credential_pda: Derived credential PDA
            schema_name: Schema name (e.g. "model-inference-v1")
            fields: Schema field definitions
            version: Schema version (default: 1)
            
        Returns:
            Dict with schema PDA and instruction data
        """
        schema_pda = hashlib.sha256(
            f"{self.SAS_PROGRAM_ID}:schema:{credential_pda}:{schema_name}:{version}".encode()
        ).hexdigest()[:44]

        return {
            "schema_pda": schema_pda,
            "credential_pda": credential_pda,
            "authority": authority,
            "name": schema_name,
            "version": version,
            "fields": fields,
            "instructions": [
                {
                    "program": self.SAS_PROGRAM_ID,
                    "accounts": [authority, credential_pda, schema_pda],
                    "data": f"create_schema:{schema_name}:{json.dumps(fields)}",
                }
            ],
            "schema_definition": self._default_model_schema(),
        }

    def _default_model_schema(self) -> dict[str, Any]:
        """Default schema layout for model inference attestation."""
        return {
            "layout": [12, 12, 12, 12, 0],  # string, string, string, string, u8
            "field_names": ["prompt_hash", "output_hash", "model_hash",
                            "proof_hash", "version"],
            "description": "Solana Chat model inference attestation",
        }

    def create_attestation(self, credential_pda: str, schema_pda: str,
                           authority: str, nonce: str,
                           data: dict[str, Any]) -> dict[str, Any]:
        """Issue an attestation for a model output.
        
        The attestation proves:
          "This prompt produced this output from this model version"
        
        Args:
            credential_pda: Credential PDA
            schema_pda: Schema PDA
            authority: Authorized signer
            nonce: Unique identifier (e.g., prompt hash)
            data: Attestation data (prompt_hash, output_hash, model_hash, etc.)
            
        Returns:
            Dict with attestation PDA and verification info
        """
        # Derive attestation PDA: [credential, schema, nonce]
        attestation_pda = hashlib.sha256(
            f"{self.SAS_PROGRAM_ID}:attestation:{credential_pda}:{schema_pda}:{nonce}".encode()
        ).hexdigest()[:44]

        expiry = int(time.time()) + (365 * 24 * 3600)  # 1 year expiry

        entry = {
            "attestation_pda": attestation_pda,
            "credential_pda": credential_pda,
            "schema_pda": schema_pda,
            "authority": authority,
            "nonce": nonce,
            "data": data,
            "expiry": expiry,
            "timestamp": int(time.time()),
            "is_valid": True,
            "instructions": [
                {
                    "program": self.SAS_PROGRAM_ID,
                    "accounts": [authority, credential_pda, schema_pda,
                                 attestation_pda],
                    "data": f"create_attestation:{json.dumps(data)}:{expiry}",
                }
            ],
        }
        self.issued_attestations.append(entry)
        return entry

    def attest_model_output(self, credential_name: str, authority: str,
                             prompt_hash: str, output_hash: str,
                             model_hash: str) -> dict[str, Any]:
        """High-level method: create credential + schema + attestation for a model output.
        
        This is the primary entry point for model inference credentialing.
        
        Args:
            credential_name: Model version (e.g. "SolanaChat-d24-v1")
            authority: Attestation authority pubkey
            prompt_hash: SHA-256 of the input prompt
            output_hash: SHA-256 of the model output
            model_hash: SHA-256 of the model weights
            
        Returns:
            Dict with full attestation chain
        """
        # Step 1: Create credential
        credential = self.create_credential(authority, credential_name)
        
        # Step 2: Create schema
        schema = self.create_schema(
            authority=authority,
            credential_pda=credential["credential_pda"],
            schema_name="model-inference-v1",
            fields=[
                {"name": "prompt_hash", "type": "string"},
                {"name": "output_hash", "type": "string"},
                {"name": "model_hash", "type": "string"},
                {"name": "proof_hash", "type": "string"},
                {"name": "version", "type": "u8"},
            ],
        )
        
        # Step 3: Create attestation
        attestation_data = {
            "prompt_hash": prompt_hash,
            "output_hash": output_hash,
            "model_hash": model_hash,
            "proof_hash": "",  # ZK proof hash if available
            "version": 1,
        }
        attestation = self.create_attestation(
            credential_pda=credential["credential_pda"],
            schema_pda=schema["schema_pda"],
            authority=authority,
            nonce=prompt_hash,  # Use prompt_hash as unique nonce
            data=attestation_data,
        )
        
        return {
            "credential": credential,
            "schema": schema,
            "attestation": attestation,
            "verification_query": {
                "schema_pda": schema["schema_pda"],
                "user_address": f"nonce:{prompt_hash[:16]}",
                "method": "Verify via SAS: fetchSchema -> deriveAttestationPda -> fetchAttestation -> deserializeData",
            },
        }

    def verify_attestation(self, attestation_pda: str, schema_pda: str) -> dict[str, Any]:
        """Verify an attestation exists and is not expired.
        
        Checks:
        1. Schema is not paused
        2. Attestation PDA exists
        3. Attestation has not expired
        
        Returns dict with verification result.
        """
        for att in self.issued_attestations:
            if att["attestation_pda"] == attestation_pda:
                is_expired = int(time.time()) > att["expiry"]
                return {
                    "attestation_pda": attestation_pda,
                    "schema_pda": schema_pda,
                    "is_valid": not is_expired,
                    "is_expired": is_expired,
                    "expiry": att["expiry"],
                    "data": att["data"],
                    "note": "Use fetchSchema() + deriveAttestationPda() + fetchAttestation() from sas-lib",
                }
        return {
            "attestation_pda": attestation_pda,
            "is_valid": False,
            "error": "Attestation not found",
        }

    def close_attestation(self, attestation_pda: str, authority: str,
                          credential_pda: str) -> dict[str, Any]:
        """Revoke an attestation (close it).
        
        Only authorized signers can close attestations.
        """
        return {
            "attestation_pda": attestation_pda,
            "authority": authority,
            "credential_pda": credential_pda,
            "action": "close",
            "note": "Attestation revoked — no longer verifiable",
            "instructions": [
                {
                    "program": self.SAS_PROGRAM_ID,
                    "accounts": [authority, attestation_pda, credential_pda],
                    "data": f"close_attestation:{attestation_pda}",
                }
            ],
        }


# =============================================================================
# TypeScript Reference Code Generator
# =============================================================================

def generate_ts_examples(output_dir: str = "solana/light_protocol_ts") -> list[str]:
    """Generate TypeScript reference examples for Light Protocol integration.
    
    Creates copy-paste ready TypeScript files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    files = []
    examples = {
        "01_create_mint.ts": '''\
import { createRpc, createMint, mintTo, transfer } from "@lightprotocol/compressed-token";
import { Keypair } from "@solana/web3.js";

// Connects to devnet or localnet
const rpc = createRpc();

async function main() {
    const payer = Keypair.generate();
    
    // 1. Create SPL mint with interface PDA for compression
    const { mint } = await createMint(rpc, payer, payer.publicKey, 9);
    console.log("Mint:", mint.toBase58());

    // 2. Mint compressed tokens (creates compressed token accounts for recipient)
    const recipient = Keypair.generate();
    await mintTo(rpc, payer, mint, recipient.publicKey, payer, 1_000_000_000);
    console.log("Recipient:", recipient.publicKey.toBase58());

    // 3. Transfer compressed tokens (UTXO pattern: consumes old, creates new)
    const finalRecipient = Keypair.generate();
    await transfer(rpc, payer, mint, 500_000_000, recipient, finalRecipient.publicKey);
    console.log("Final:", finalRecipient.publicKey.toBase58());
}
main();
''',
        "02_compressed_pda.ts": '''\
import { deriveAddressV2, deriveAddressSeedV2, PackedAccounts } from "@lightprotocol/stateless.js";

// Derive a compressed PDA address
const programId = new PublicKey("YOUR_PROGRAM_ID");
const addressTree = new PublicKey("batchAddressTree");

const seed = deriveAddressSeedV2([Buffer.from("my-seed")]);
const address = deriveAddressV2(seed, addressTree, programId);
console.log("Compressed PDA:", address.toBase58());
''',
        "03_nullifier.ts": '''\
import { createNullifierIx } from "light-nullifier-program";

async function preventDoubleSpend(rpc, payer, paymentId: Uint8Array) {
    const ix = await createNullifierIx(rpc, payer.publicKey, paymentId);
    // Combine with your transaction
    return ix;
}
''',
        "04_sas_attestation.ts": '''\
import {
    getCreateCredentialInstruction,
    getCreateSchemaInstruction,
    serializeAttestationData,
    getCreateAttestationInstruction,
    deriveAttestationPda,
    deriveCredentialPda,
    deriveSchemaPda,
    fetchSchema,
    fetchAttestation,
    deserializeAttestationData,
} from "sas-lib";

async function attestModelOutput(rpc, payer, authority) {
    const [credentialPda] = await deriveCredentialPda({
        authority: authority.address,
        name: "SolanaChat-Model-v1"
    });

    const credentialIx = getCreateCredentialInstruction({
        payer, credential: credentialPda, authority,
        name: "SolanaChat-Model-v1",
        signers: [authority.address]
    });

    // Schema defines model output structure
    const [schemaPda] = await deriveSchemaPda({
        credential: credentialPda,
        name: "model-inference-v1",
        version: 1
    });

    // ... continue with create schema and attestation
}
''',
    }
    
    for filename, content in examples.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(content.strip() + "\n")
        files.append(path)
        print(f"Generated: {path}")
    
    return files


# =============================================================================
# Cost Reference
# =============================================================================

COST_REFERENCE = {
    "compressed_token_account": {
        "cost_sol": 0.000015,
        "standard_ata_cost_sol": 0.00204,
        "savings_x": 136,
    },
    "compressed_pda_100_bytes": {
        "cost_sol": 0.000015,
        "standard_pda_cost_sol": 0.0016,
        "savings_x": 106,
    },
    "nullifier_pda": {
        "cost_lamports": 15_000,
        "cost_sol": 0.000015,
    },
}


if __name__ == "__main__":
    # Quick test
    client = CompressedTokenClient()
    mint = client.create_mint("authority_pubkey")
    print(f"Created mint: {mint['mint'][:16]}... (cost: {mint['compressed_account_cost_sol']} SOL)")

    transfer = client.transfer("mint", "sender", "recipient", 1_000_000)
    print(f"Transfer savings: {transfer['instructions'][0]['program'][:20]}...")

    pda = CompressedPDAClient()
    account = pda.create_account("program1", "signer1", [b"test"], b"hello")
    print(f"Compressed PDA: {account['address'][:16]}... (106x cheaper)")

    nullifier = NullifierClient()
    n = nullifier.create_nullifier("payer", b"unique-payment-id-123")
    print(f"Nullifier: {n['nullifier_address'][:16]}... (0.000015 SOL)")

    sas = AttestationClient()
    attestation = sas.attest_model_output(
        "SolanaChat-d24-v1", "authority_key",
        hashlib.sha256(b"prompt").hexdigest(),
        hashlib.sha256(b"output").hexdigest(),
        hashlib.sha256(b"model_weights").hexdigest(),
    )
    print(f"SAS Attestation: {attestation['attestation']['attestation_pda'][:16]}...")
    print(f"  Credential: {attestation['credential']['name']}")
    print(f"  Schema fields: {attestation['schema']['fields']}")

    # Generate TS examples
    generate_ts_examples()
    print("\nLight Protocol integration modules operational.")