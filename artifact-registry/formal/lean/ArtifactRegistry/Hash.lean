import ArtifactRegistry.Types

namespace ArtifactRegistry

/-- SkillHub merkle-leaf preimage: `slug || 0x00 || bundleHash`
    (bundleHash includes the `sha256-` prefix). -/
def skillLeafPreimage (slug bundleHash : String) : String :=
  slug ++ String.singleton (Char.ofNat 0) ++ bundleHash

/-- Tagged canonical preimage: `kind || 0x00 || canonicalJson`. -/
def taggedPreimage (kind canonicalJson : String) : String :=
  kind ++ String.singleton (Char.ofNat 0) ++ canonicalJson

theorem skillLeafPreimage_def (slug bundleHash : String) :
    skillLeafPreimage slug bundleHash =
      slug ++ String.singleton (Char.ofNat 0) ++ bundleHash := rfl

theorem taggedPreimage_def (kind body : String) :
    taggedPreimage kind body =
      kind ++ String.singleton (Char.ofNat 0) ++ body := rfl


def hexDigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat ('0'.toNat + n)
  else Char.ofNat ('a'.toNat + (n - 10))

def byteHex (n : Nat) : String :=
  String.singleton (hexDigit ((n / 16) % 16)) ++
    String.singleton (hexDigit (n % 16))

def toHex (s : String) : String :=
  String.join (s.toList.map (fun c => byteHex c.toNat))

end ArtifactRegistry
