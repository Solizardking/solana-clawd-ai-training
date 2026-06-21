const state = {
  lane: document.getElementById("lane"),
  inputs: document.getElementById("inputs"),
  datasetRepo: document.getElementById("datasetRepo"),
  modelRepo: document.getElementById("modelRepo"),
  pushDataset: document.getElementById("pushDataset"),
  train: document.getElementById("train"),
  remoteTrain: document.getElementById("remoteTrain"),
  pushModel: document.getElementById("pushModel"),
  register: document.getElementById("register"),
  liveRegister: document.getElementById("liveRegister"),
  yes: document.getElementById("yes"),
  output: document.getElementById("commandOutput"),
};

const lanes = {
  custom: {
    datasetRepo: "solanaclawd/solana-clawd-realtime-research-instruct",
    modelRepo: "solanaclawd/solana-clawd-custom-lora",
  },
  "core-ai": {
    datasetRepo: "solanaclawd/solana-clawd-core-ai-instruct",
    modelRepo: "solanaclawd/solana-clawd-core-ai-1.5b-lora",
  },
  "trading-factory": {
    datasetRepo: "solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
    modelRepo: "solanaclawd/solana-nvidia-trading-factory-8b-lora",
  },
  perps: {
    datasetRepo: "solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
    modelRepo: "solanaclawd/solana-clawd-perps-tools-lora",
  },
  "tx-foundation": {
    datasetRepo: "solanaclawd/solana-tx-foundation-cpt",
    modelRepo: "solanaclawd/solana-tx-foundation-1.5b",
  },
};

function shellQuote(value) {
  if (/^[A-Za-z0-9_./:@=-]+$/.test(value)) return value;
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function inputArgs() {
  return state.inputs.value
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map(shellQuote)
    .join(" ");
}

function buildCommand() {
  const lane = state.lane.value;
  const parts = [
    "ai-training/model-kit/bin/clawd-model-kit",
    "one-shot",
    inputArgs(),
    "--lane",
    lane,
    "--dataset-repo",
    shellQuote(state.datasetRepo.value.trim()),
    "--hub-model-id",
    shellQuote(state.modelRepo.value.trim()),
    "--output-prefix",
    "data/model_kit/model_kit",
  ].filter(Boolean);

  if (state.pushDataset.checked) parts.push("--push-dataset");
  if (state.train.checked) parts.push("--train");
  if (state.remoteTrain.checked) parts.push("--remote-train");
  if (state.pushModel.checked) parts.push("--push-model");
  if (state.register.checked) parts.push("--register");
  if (state.liveRegister.checked) parts.push("--live-register");
  if (state.yes.checked) parts.push("--yes");

  const lines = [
    "cd /Users/8bit/Downloads/solana-clawd",
    parts.join(" \\\n  "),
    "",
    "# Local checks",
    "ai-training/model-kit/bin/clawd-model-kit doctor",
    "ai-training/model-kit/bin/clawd-model-kit verify",
    "",
    "# Registry dry-run",
    `ai-training/model-kit/bin/clawd-model-kit register --hf-model ${shellQuote(state.modelRepo.value.trim())}`,
    "",
    "# Perps tool lane",
    "ai-training/model-kit/bin/clawd-model-kit perps tools --write --json",
    "ai-training/model-kit/bin/clawd-model-kit perps handoff --market SOL --mode observer",
  ];
  state.output.textContent = lines.join("\n");
}

state.lane.addEventListener("change", () => {
  const next = lanes[state.lane.value];
  state.datasetRepo.value = next.datasetRepo;
  state.modelRepo.value = next.modelRepo;
  buildCommand();
});

document.getElementById("generate").addEventListener("click", buildCommand);

document.getElementById("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.output.textContent);
});

[
  state.inputs,
  state.datasetRepo,
  state.modelRepo,
  state.pushDataset,
  state.train,
  state.remoteTrain,
  state.pushModel,
  state.register,
  state.liveRegister,
  state.yes,
].forEach((element) => element.addEventListener("input", buildCommand));

buildCommand();
