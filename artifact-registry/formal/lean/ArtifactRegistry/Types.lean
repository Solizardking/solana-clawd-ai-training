/-
  Type-generic artifact registry entries.
  Mirrors the TypeScript module in ../../src and the Musebook agent-registry
  record shape, generalized to agent | model | skill | tool | mcp_server.
-/

namespace ArtifactRegistry

inductive EntryType
  | agent
  | model
  | skill
  | tool
  | mcpServer
  deriving DecidableEq, Repr, Inhabited

def EntryType.toName : EntryType → String
  | .agent => "agent"
  | .model => "model"
  | .skill => "skill"
  | .tool => "tool"
  | .mcpServer => "mcp_server"

structure Attestation where
  kind : String
  zkPending : Bool
  vkSlotsEmpty : Bool
  programDeployed : Bool
  deriving DecidableEq, Repr

/-- Until circuits exist, verifying-key slots stay empty and the program is
    not deployed, so attestation must remain pending. -/
def attestationValid (a : Attestation) : Prop :=
  (a.programDeployed = false ∧ a.vkSlotsEmpty = true) → a.zkPending = true

def pendingAttestation : Attestation :=
  { kind := "authority-signature"
    zkPending := true
    vkSlotsEmpty := true
    programDeployed := false }

structure Record where
  type : EntryType
  contentHash : String
  version : String
  authority : String
  /-- True when contentHash equals the canonical hash of the bound payload. -/
  payloadBound : Bool
  clusterDevnet : Bool
  attestation : Attestation
  deriving DecidableEq, Repr

/-- The uniqueness key formalized here: (type, content hash, version). -/
def key (r : Record) : EntryType × String × String :=
  (r.type, r.contentHash, r.version)

def recordValid (r : Record) : Prop :=
  r.clusterDevnet = true ∧ r.payloadBound = true ∧ r.version ≠ "" ∧ r.authority ≠ "" ∧
    attestationValid r.attestation

/-- A registry is a list of records with unique keys. -/
def uniqueKeys (xs : List Record) : Prop :=
  xs.Pairwise (fun a b => key a ≠ key b)

end ArtifactRegistry
