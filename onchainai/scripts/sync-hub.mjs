import { writeFile } from 'node:fs/promises';
const repos = {};
for (const kind of ['models', 'datasets', 'spaces']) {
  const responses = await Promise.all(['solanaclawd', 'ordlibrary'].map(async author => {
    const response = await fetch(`https://huggingface.co/api/${kind}?author=${author}&limit=1000&full=true`, { signal: AbortSignal.timeout(20000) });
    if (!response.ok) throw new Error(`Hub ${kind}: HTTP ${response.status}`);
    return response.json();
  }));
  repos[kind] = responses.flat().map(item => ({
    id: item.id, kind: kind.slice(0, -1), sha: item.sha || null,
    downloads: item.downloads || 0, likes: item.likes || 0,
    pipeline_tag: item.pipeline_tag || null, tags: item.tags || [],
    last_modified: item.lastModified || null,
    url: `https://huggingface.co/${kind === 'models' ? '' : `${kind}/`}${item.id}`,
    files: (item.siblings || []).map(file => file.rfilename),
    card: { license: item.cardData?.license || null, base_model: item.cardData?.base_model || null },
  }));
}
await writeFile(new URL('../hub-catalog.json', import.meta.url), JSON.stringify({ updated_at: new Date().toISOString(), source: 'Hugging Face public APIs: solanaclawd and ordlibrary', ...repos }, null, 2));
console.log(`Synced ${repos.models.length} models, ${repos.datasets.length} datasets, ${repos.spaces.length} Spaces`);
