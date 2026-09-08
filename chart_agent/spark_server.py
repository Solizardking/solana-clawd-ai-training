"""Single-GPU Spark service. Start explicitly with uvicorn chart_agent.spark_server:app."""
import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from .spark_protocol import parse_tool_calls

MODEL = 'XHToken/Spark-X2.5-4B'
REVISION = '5e10fcc0286756aebf7c41dc52c1e42d95c70281'
CODE = {'configuration_spark.py':'02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722', 'modeling_spark.py':'9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4'}
state = {}
lock = asyncio.Lock()


def load():
    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Spark service requires a CUDA GPU')
    root = Path(snapshot_download(MODEL, revision=REVISION))
    for name, digest in CODE.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Spark custom code hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='eager').eval()
    adapter = os.getenv('SPARK_ADAPTER_REPO')
    if adapter:
        revision = os.getenv('SPARK_ADAPTER_REVISION', '')
        if not re.fullmatch('[a-f0-9]{40}', revision):
            raise RuntimeError('Pin SPARK_ADAPTER_REVISION to a full commit')
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, revision=revision).eval()
    state.update(model=model, tokenizer=tokenizer, adapter=adapter, alias=os.getenv('CHART_MODEL_NAME', 'clawd-spark'))


@asynccontextmanager
async def lifespan(app):
    if not os.getenv('CHART_MODEL_API_KEY'):
        raise RuntimeError('CHART_MODEL_API_KEY required')
    await asyncio.to_thread(load)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


async def auth(authorization: str = Header(default=''), x_chart_model_key: str = Header(default='')):
    key = os.getenv('CHART_MODEL_API_KEY', '')
    if not key or not secrets.compare_digest(x_chart_model_key or (authorization[7:] if authorization.startswith('Bearer ') else ''), key):
        raise HTTPException(401, 'Model bearer key required')


@app.get('/health', dependencies=[Depends(auth)])
async def health():
    if not state:
        raise HTTPException(503, 'Model not loaded')
    return {'model':state['alias'], 'adapter':state['adapter'], 'base_revision':REVISION}


class TokenInput(BaseModel):
    model: str
    prompt: str = Field(max_length=32000)
    add_special_tokens: bool = False


class Tokens(BaseModel):
    model: str
    tokens: list[int] = Field(max_length=32000)


def check_model(name):
    if name != state.get('alias'):
        raise HTTPException(404, 'Unknown model')


@app.post('/tokenize', dependencies=[Depends(auth)])
async def tokenize(body: TokenInput):
    check_model(body.model)
    ids = state['tokenizer'].encode(body.prompt, add_special_tokens=body.add_special_tokens)
    return {'tokens':ids, 'count':len(ids)}


@app.post('/detokenize', dependencies=[Depends(auth)])
async def detokenize(body: Tokens):
    check_model(body.model)
    if any(i < 0 or i >= len(state['tokenizer']) for i in body.tokens):
        raise HTTPException(422, 'Invalid token id')
    return {'prompt':state['tokenizer'].decode(body.tokens)}


class Chat(BaseModel):
    model: str
    messages: list[dict] = Field(min_length=1, max_length=40)
    tools: list[dict] = Field(default_factory=list, max_length=32)
    tool_choice: str = 'auto'
    max_tokens: int = Field(default=512, ge=1, le=2400)
    temperature: float = Field(default=0, ge=0, le=2)
    stream: bool = False
    chat_template_kwargs: dict = Field(default_factory=dict)


def generate(body):
    import torch
    tokenizer, model = state['tokenizer'], state['model']
    messages = copy.deepcopy(body.messages)
    for message in messages:
        for call in message.get('tool_calls') or []:
            if isinstance(call['function']['arguments'], str):
                call['function']['arguments'] = json.loads(call['function']['arguments'])
    tools = body.tools if body.tool_choice == 'auto' else []
    prompt = tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    count = inputs['input_ids'].shape[-1]
    if count + body.max_tokens > int(os.getenv('SPARK_CONTEXT_SIZE', '4096')):
        raise ValueError('Prompt and requested output exceed configured context capacity')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=body.max_tokens, do_sample=body.temperature>0,
                                **({'temperature':body.temperature, 'top_k':0} if body.temperature>0 else {}), logits_to_keep=1, pad_token_id=tokenizer.pad_token_id)
    ids = output[0,count:].tolist()
    finish = 'stop' if ids and ids[-1] == tokenizer.eos_token_id else 'length'
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # Never parse or execute a truncated native request.
    if '<tool_call>' in text and finish == 'length':
        raise ValueError('Truncated Spark tool request')
    calls = parse_tool_calls(text, tools)
    message = {'role':'assistant', 'content':None if calls else text}
    if calls:
        message['tool_calls'] = calls
        finish = 'tool_calls'
    return {'model':body.model, 'choices':[{'index':0, 'message':message, 'finish_reason':finish}],
            'usage':{'prompt_tokens':count, 'completion_tokens':len(ids), 'total_tokens':count+len(ids)}}


@app.post('/v1/chat/completions', dependencies=[Depends(auth)])
async def chat(body: Chat):
    check_model(body.model)
    if body.stream or body.tool_choice not in ('auto', 'none'):
        raise HTTPException(422, 'Use non-streaming chat and auto/none tool choice')
    if lock.locked():
        raise HTTPException(429, 'Model busy')
    async with lock:
        try:
            return await asyncio.to_thread(generate, body)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
