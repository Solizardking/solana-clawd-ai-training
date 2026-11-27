import ArtifactRegistry.Types

namespace ArtifactRegistry

theorem pendingAttestation_valid : attestationValid pendingAttestation := by
  intro _; rfl

theorem vk_empty_undeployed_implies_pending
    (a : Attestation) (h : a.programDeployed = false ∧ a.vkSlotsEmpty = true)
    (hv : attestationValid a) :
    a.zkPending = true :=
  hv h

theorem record_valid_requires_devnet (r : Record) (h : recordValid r) : r.clusterDevnet = true :=
  h.1

theorem record_valid_requires_binding (r : Record) (h : recordValid r) : r.payloadBound = true :=
  h.2.1

theorem record_valid_requires_pending_when_vk_empty
    (r : Record) (h : recordValid r)
    (hv : r.attestation.programDeployed = false ∧ r.attestation.vkSlotsEmpty = true) :
    r.attestation.zkPending = true :=
  h.2.2.2.2 hv

theorem uniqueKeys_nil : uniqueKeys ([] : List Record) := by
  simp [uniqueKeys]

theorem uniqueKeys_cons {r : Record} {xs : List Record}
    (hxs : uniqueKeys xs)
    (hfresh : ∀ s, s ∈ xs → key s ≠ key r) :
    uniqueKeys (r :: xs) := by
  unfold uniqueKeys at hxs ⊢
  exact List.pairwise_cons.mpr ⟨fun s hs => (hfresh s hs).symm, hxs⟩

theorem uniqueKeys_reject_duplicate
    {xs : List Record} {r s : Record}
    (huniq : uniqueKeys (r :: xs)) (hin : s ∈ xs) :
    key s ≠ key r := by
  unfold uniqueKeys at huniq
  have hrel := (List.pairwise_cons.mp huniq).1
  exact (hrel s hin).symm

/-- Authority is part of the record, but the uniqueness key does not include it:
    a second authority cannot occupy the same (type, hash, version). -/
theorem uniqueness_key_ignores_authority
    (r₁ r₂ : Record)
    (ht : r₁.type = r₂.type)
    (hh : r₁.contentHash = r₂.contentHash)
    (hv : r₁.version = r₂.version) :
    key r₁ = key r₂ := by
  cases r₁; cases r₂; simp [key] at *
  exact ⟨ht, hh, hv⟩

end ArtifactRegistry
