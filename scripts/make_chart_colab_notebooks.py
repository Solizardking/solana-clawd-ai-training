#!/usr/bin/env python3
"""Create validated Colab training and GGUF vision notebooks from project scripts."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def write(name, cells, gpu):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "colab": {"name": name, "gpuType": gpu}, "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source, name, "exec")
    nbf.write(notebook, OUT / name)
    print(OUT / name)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
trainer = (ROOT / "scripts/train_chart_foundation.py").read_text()
write("clawd_chart_foundation_27b_train.ipynb", [
    md("# Clawd Chart Foundation 27B\n\nTrain a new LoRA adapter on the trainable DavidAU base behind your GGUF release. Two stages: document adaptation, then text/chart-image supervised training. **No model has been trained by generating this notebook.** Select an A100 (40 GB+) runtime before running.\n\nThe supplied Colab link is an inference-only demo. This notebook is the separate training workflow. Your current Colab account had A100 disabled during setup; a T4 cannot run this recipe."),
    md("## 1. Check the runtime"),
    code("import torch\nassert torch.cuda.is_available(), 'Select an A100 or larger GPU runtime.'\nprint(torch.cuda.get_device_name(0))\nassert torch.cuda.get_device_properties(0).total_memory >= 39 * 1024**3, 'At least 40 GB VRAM required for this recipe.'"),
    md("## 2. Install the training tools\nThe installed package versions are recorded with the resulting adapter."),
    code("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers==5.16.1', 'peft==0.20.0', 'accelerate==1.14.0', 'bitsandbytes>=0.46,<1', 'pillow==12.3.0'], check=True)"),
    md("## 3. Upload the prepared data archive\nUpload `outputs/chart-foundation-data.zip` from the project. This contains the local datasets, chart images, paper text, and a bounded historical Solarchive sample. Mac paths are not accessible inside Colab. Dataset cards/manifests and unsupported preference records are inventoried rather than mislabeled as supervised examples. Holdouts are kept separate."),
    code("from pathlib import Path\nfrom google.colab import files\nimport zipfile, json\narchive = Path('/content/chart-foundation-data.zip')\nif not archive.exists():\n    files.upload()\nassert archive.is_file(), 'Upload chart-foundation-data.zip'\ndata = Path('/content/chart-foundation-data')\ndata.mkdir(exist_ok=True)\nwith zipfile.ZipFile(archive) as z:\n    for item in z.infolist():\n        assert (data / item.filename).resolve().is_relative_to(data.resolve()), 'Unsafe archive path'\n    z.extractall(data)\nmanifest = json.loads((data / 'manifest.json').read_text())\nprint(manifest['counts'])\nprint(manifest['limitations'])"),
    md("## 4. Authenticate privately\nUse Colab Secrets for `HF_TOKEN`, or an existing Hugging Face login. Do not put tokens in notebook cells."),
    code("import os\nfrom google.colab import userdata\ntry:\n    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\nexcept (userdata.SecretNotFoundError, userdata.NotebookAccessError):\n    print('No Colab HF_TOKEN available; public model access will be attempted.')"),
    md("## 5. Validate and train\nThe language adapter uses full-sequence loss, with padding and vision markers masked. Vision encoder weights are frozen. Document training precedes supervised training; held-out validation/test records never enter either training split. Default is one complete epoch per stage. Setting a positive step cap creates a limited pilot, not a full run."),
    code("from pathlib import Path\nPath('/content/train_chart_foundation.py').write_text(" + repr(trainer) + ")\nsubprocess.run([sys.executable, '/content/train_chart_foundation.py', '--data', str(data), '--preflight'], check=True)"),
    code("MAX_STEPS = -1  # -1 = full epoch per stage; a positive value is a pilot run\noutput = Path('/content/clawd-chart-foundation-27b-lora')\nsubprocess.run([sys.executable, '/content/train_chart_foundation.py', '--data', str(data), '--output', str(output), '--max-steps', str(MAX_STEPS)], check=True)\n(output / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))"),
    md("## 6. Review and download the result\nThe saved artifact is a new adapter, not a merged GGUF. It must be loaded with the pinned trainable base. Evaluation loss measures language modeling fit; it does not establish trading performance. No automatic publication or live trading occurs."),
    code("import shutil\nprint(json.loads((output / 'metrics.json').read_text()))\nassert (output / 'adapter/adapter_model.safetensors').is_file()\narchive_out = shutil.make_archive('/content/clawd-chart-foundation-27b-lora', 'zip', output)\nfiles.download(archive_out)"),
], "A100")

inference = (ROOT / "scripts/davidau_gguf_vision.py").read_text()
write("davidau_qwen_gguf_vision.ipynb", [
    md("# DavidAU Qwen GGUF vision\n\nCorrected version of the supplied [Colab inference demo](https://colab.research.google.com/notebook#fileId=https%3A//huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF.ipynb). Uses the exact IQ2_M GGUF plus `mmproj-F16.gguf` and the upstream `MTMDChatHandler`. This runs the existing model; it does not create or train a new model.\n\nCode/schema validated locally. GPU execution and image quality remain unverified. Requires a current CUDA-enabled llama-cpp-python build supporting this architecture; installation compiles native code and downloads multi-GB weights in Colab."),
    md("## 1. Build the CUDA backend"),
    code("import os, subprocess, sys\nsubprocess.run(['nvidia-smi'], check=True)\nos.environ['CMAKE_ARGS'] = '-DGGML_CUDA=on'\nos.environ['FORCE_CMAKE'] = '1'\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', 'git+https://github.com/abetlen/llama-cpp-python.git', 'huggingface_hub'], check=True)\nfrom llama_cpp.llama_chat_format import MTMDChatHandler\nprint('MTMD handler available')"),
    md("## 2. Load the model and projector, then describe an image\nUse a plain HTTPS URL, not Markdown link syntax. Replace the default image with a chart image or a Colab-local file path."),
    code("from pathlib import Path\nPath('/content/davidau_gguf_vision.py').write_text(" + repr(inference) + ")\nIMAGE = 'https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg'\nPROMPT = 'Describe this image in one sentence.'\nsubprocess.run([sys.executable, '/content/davidau_gguf_vision.py', '--image', IMAGE, '--prompt', PROMPT], check=True)"),
    md("## 3. Record the environment"),
    code("Path('/content/gguf-environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))"),
], "T4")
