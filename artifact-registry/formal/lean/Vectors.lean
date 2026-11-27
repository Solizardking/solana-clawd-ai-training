import ArtifactRegistry

open ArtifactRegistry

def skillhubBundleHash : String :=
  "sha256-f393b771a8b5fdfb2665f9eaa1a1205dc9fdf6ed1b118819527152a4d1493c46"

def skillhubSlug : String :=
  "skillhub-onchain"

def hauhauOnchainCanonical : String :=
  "{\"files\":[{\"path\":\".gitattributes\",\"sha256\":\"48b460424f20e8c7494e03bb2e96e3ce8591c402e929d6fc8244367b9180d5a2\",\"size\":1673},{\"path\":\"Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf\",\"sha256\":\"ba3a1d47a604f17ef74d913f9a9d9a2b456acc6925e7b168bfa2e6527246011c\",\"size\":11659235456},{\"path\":\"README.md\",\"sha256\":\"d27e32be97c4099105b817c383f1ca54cc399e0e015ace994374f1f34706cd74\",\"size\":23979}],\"hfRepo\":\"ordlibrary/hauhau-qwen36-onchain\",\"license\":\"mit\",\"provenance\":{\"docs\":[{\"path\":\"docs/hauhau_qwen36.md\",\"sha256\":\"sha256-72bd77af7cf040194aac41a81a64a6dfacc7af587d9cd930def0e0caa89eecac\"},{\"path\":\"configs/hauhau_qwen36_llama_cpp.yaml\",\"sha256\":\"sha256-86f51584f059a50e0bbea3e61d6823b8dc42ed0486f40952cad42678fe486510\"},{\"path\":\"ollama/Modelfile.hauhau-qwen36-onchain\",\"sha256\":\"sha256-0f468ebf13f67558cf021fad6bdf185b48129c7006323a4d8f526e5d4ac01532\"}],\"note\":\"Constitution-etched GGUF runtime documented in this training repo. GGUF hash is the Hugging Face LFS SHA-256; other files are SHA-256 of the bytes at this revision.\",\"sourceRepo\":\"Solizardking/solana-clawd-ai-training\"},\"revisionSha\":\"aeb339dc7de361b468683b1242729c701cba465e\"}"

def cheshireToolsCanonical : String :=
  "{\"commit\":\"be3de0244dc1a57e323d8160787ebb40fb64930e\",\"manifestHash\":\"a2977cc1558760d3c42528e1d2b8345440d35aa6902705064db7c2e57e40e6fe\",\"name\":\"cheshire-mcp-tools\",\"sourceRepo\":\"Solizardking/skillhub-main\"}"

def cheshireMcpCanonical : String :=
  "{\"commit\":\"be3de0244dc1a57e323d8160787ebb40fb64930e\",\"endpoint\":\"https://cheshireterminal.ai/mcp\",\"manifestHash\":\"c326b74073742aaa77c849effabb0ea8eb474efbab905bba6f7590eaabc02004\",\"name\":\"cheshire-terminal\",\"packageVersion\":\"2026-07-21\",\"sourceRepo\":\"Solizardking/skillhub-main\"}"

def skillhubLeafPreimage : String :=
  skillLeafPreimage skillhubSlug skillhubBundleHash

def hauhauOnchainPreimage : String :=
  taggedPreimage "model-v1" hauhauOnchainCanonical

def cheshireToolsPreimage : String :=
  taggedPreimage "tool-v1" cheshireToolsCanonical

def cheshireMcpPreimage : String :=
  taggedPreimage "mcp_server-v1" cheshireMcpCanonical

/-- Emit JSON array of named UTF-8 hex preimages for the TypeScript parity test. -/
def emit : String :=
  let entry (name s : String) :=
    "{\"name\":\"" ++ name ++ "\",\"preimageHex\":\"" ++ toHex s ++ "\"}"
  "[" ++
    String.intercalate "," [
      entry "skillhub_onchain_leaf_preimage" skillhubLeafPreimage,
      entry "hauhau_onchain_model_preimage" hauhauOnchainPreimage,
      entry "cheshire_tools_preimage" cheshireToolsPreimage,
      entry "cheshire_mcp_preimage" cheshireMcpPreimage
    ] ++
  "]"

def main : IO Unit := do
  IO.println emit
