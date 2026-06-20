/**
 * ClawVault — Persistent Markdown Memory System for ClawdBot
 *
 * Architecture: Session → Observe → Score → Route → Store → Reflect → Promote
 *
 * Vault structure:
 *   vault/decisions/   — trade decisions with rationale
 *   vault/lessons/     — learned patterns and insights
 *   vault/trades/      — trade outcomes and P&L
 *   vault/research/    — auto-research experiment logs
 *   vault/tasks/       — pending agent tasks
 *   vault/backlog/     — deferred items
 *   vault/inbox/       — raw incoming observations
 *
 * Internal state (.clawvault/):
 *   graph-index.json   — cross-document link graph
 *   last-checkpoint.json — wake/sleep state
 *   config.json        — vault configuration
 */

import fs from "fs/promises";
import path from "path";
import matter from "gray-matter";
import { format, parseISO } from "date-fns";

export type VaultCategory =
  | "decisions"
  | "lessons"
  | "trades"
  | "research"
  | "tasks"
  | "backlog"
  | "inbox";

export interface VaultEntry {
  id: string;
  category: VaultCategory;
  title: string;
  content: string;
  tags: string[];
  links: string[]; // other entry IDs
  score: number; // relevance/importance 0-1
  createdAt: string;
  updatedAt: string;
  metadata: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  category: VaultCategory;
  title: string;
  tags: string[];
  score: number;
  links: string[];
  path: string;
}

export interface GraphIndex {
  nodes: Record<string, GraphNode>;
  edges: Array<{ from: string; to: string; weight: number }>;
  lastUpdated: string;
}

export interface Checkpoint {
  sessionId: string;
  agentState: Record<string, unknown>;
  activePositions: unknown[];
  pendingResearch: string[];
  lastObservation: string;
  memory: {
    shortTerm: string[]; // recent entry IDs
    promotedIds: string[]; // promoted to long-term
  };
  createdAt: string;
}

// ── Scoring heuristics for auto-routing ─────────────────────────────
const CATEGORY_KEYWORDS: Record<VaultCategory, string[]> = {
  decisions: ["decided", "chose", "selected", "bought", "sold", "entered", "exited"],
  lessons: ["learned", "realized", "insight", "pattern", "mistake", "always", "never"],
  trades: ["pnl", "profit", "loss", "position", "entry", "exit", "size", "fee"],
  research: ["hypothesis", "experiment", "result", "metric", "strategy", "backtest"],
  tasks: ["todo", "need to", "should", "must", "action", "implement", "fix"],
  backlog: ["later", "eventually", "someday", "consider", "idea"],
  inbox: [],
};

export class ClawVault {
  private vaultPath: string;
  private clawvaultPath: string;
  private graphIndex: GraphIndex;
  private shortTermBuffer: VaultEntry[] = [];
  private readonly SHORT_TERM_MAX = 50;

  constructor(vaultPath: string = "./vault") {
    this.vaultPath = path.resolve(vaultPath);
    this.clawvaultPath = path.join(this.vaultPath, "..", ".clawvault");
    this.graphIndex = { nodes: {}, edges: [], lastUpdated: new Date().toISOString() };
  }

  // ── Init ────────────────────────────────────────────────────────────

  async init(): Promise<void> {
    const categories: VaultCategory[] = [
      "decisions", "lessons", "trades", "research", "tasks", "backlog", "inbox",
    ];

    for (const cat of categories) {
      await fs.mkdir(path.join(this.vaultPath, cat), { recursive: true });
    }
    await fs.mkdir(this.clawvaultPath, { recursive: true });

    // Load or create graph index
    const graphPath = path.join(this.clawvaultPath, "graph-index.json");
    try {
      const raw = await fs.readFile(graphPath, "utf-8");
      this.graphIndex = JSON.parse(raw) as GraphIndex;
    } catch {
      await this.saveGraphIndex();
    }

    console.log("🗄️  ClawVault initialized");
  }

  // ── Store ────────────────────────────────────────────────────────────

  /**
   * !remember — Store knowledge in vault
   * Routes content to appropriate category based on scoring
   */
  async remember(
    content: string,
    opts: {
      category?: VaultCategory;
      title?: string;
      tags?: string[];
      metadata?: Record<string, unknown>;
      score?: number;
    } = {}
  ): Promise<VaultEntry> {
    const category = opts.category ?? this.autoRoute(content);
    const id = this.generateId(category);
    const now = new Date().toISOString();

    const entry: VaultEntry = {
      id,
      category,
      title: opts.title ?? this.extractTitle(content),
      content,
      tags: opts.tags ?? this.extractTags(content),
      links: [],
      score: opts.score ?? this.scoreContent(content),
      createdAt: now,
      updatedAt: now,
      metadata: opts.metadata ?? {},
    };

    // Write to markdown file
    const filePath = this.entryPath(entry);
    const frontmatter = {
      id,
      category,
      title: entry.title,
      tags: entry.tags,
      score: entry.score,
      createdAt: entry.createdAt,
      updatedAt: entry.updatedAt,
      ...entry.metadata,
    };

    const fileContent = matter.stringify(content, frontmatter);
    await fs.writeFile(filePath, fileContent, "utf-8");

    // Update graph index
    this.graphIndex.nodes[id] = {
      id,
      category,
      title: entry.title,
      tags: entry.tags,
      score: entry.score,
      links: [],
      path: filePath,
    };
    await this.saveGraphIndex();

    // Add to short-term buffer
    this.shortTermBuffer.push(entry);
    if (this.shortTermBuffer.length > this.SHORT_TERM_MAX) {
      this.shortTermBuffer.shift();
    }

    return entry;
  }

  // ── Retrieve ─────────────────────────────────────────────────────────

  /**
   * !recall — Retrieve relevant memories
   * Combines keyword search + tag matching + graph traversal
   */
  async recall(
    query: string,
    opts: {
      category?: VaultCategory;
      limit?: number;
      minScore?: number;
    } = {}
  ): Promise<VaultEntry[]> {
    const limit = opts.limit ?? 10;
    const allEntries = await this.loadAllEntries(opts.category);

    // Score each entry against query
    const scored = allEntries.map((entry) => ({
      entry,
      relevance: this.computeRelevance(query, entry),
    }));

    // Sort by relevance × importance
    scored.sort((a, b) => b.relevance * b.entry.score - a.relevance * a.entry.score);

    return scored
      .filter((s) => s.relevance > (opts.minScore ?? 0.1))
      .slice(0, limit)
      .map((s) => s.entry);
  }

  /**
   * Short-term context — recent entries without disk IO
   */
  getShortTermContext(limit = 10): VaultEntry[] {
    return this.shortTermBuffer.slice(-limit);
  }

  // ── Graph Traversal ──────────────────────────────────────────────────

  async linkEntries(fromId: string, toId: string, weight = 1.0): Promise<void> {
    const fromNode = this.graphIndex.nodes[fromId];
    const toNode = this.graphIndex.nodes[toId];
    if (!fromNode || !toNode) return;

    if (!fromNode.links.includes(toId)) {
      fromNode.links.push(toId);
    }

    this.graphIndex.edges.push({ from: fromId, to: toId, weight });
    await this.saveGraphIndex();

    // Update file frontmatter
    const filePath = fromNode.path;
    try {
      const raw = await fs.readFile(filePath, "utf-8");
      const parsed = matter(raw);
      const links: string[] = (parsed.data.links as string[]) ?? [];
      if (!links.includes(toId)) links.push(toId);
      parsed.data.links = links;
      await fs.writeFile(filePath, matter.stringify(parsed.content, parsed.data), "utf-8");
    } catch {
      // File may have been deleted
    }
  }

  async traverseGraph(startId: string, depth = 2): Promise<VaultEntry[]> {
    const visited = new Set<string>();
    const results: VaultEntry[] = [];

    const traverse = async (id: string, currentDepth: number): Promise<void> => {
      if (visited.has(id) || currentDepth < 0) return;
      visited.add(id);

      const node = this.graphIndex.nodes[id];
      if (!node) return;

      try {
        const entry = await this.loadEntry(node.path);
        if (entry) results.push(entry);

        for (const linkedId of node.links) {
          await traverse(linkedId, currentDepth - 1);
        }
      } catch {
        // Node file missing
      }
    };

    await traverse(startId, depth);
    return results;
  }

  // ── Checkpoint (Wake/Sleep) ──────────────────────────────────────────

  async saveCheckpoint(state: Omit<Checkpoint, "createdAt" | "memory">): Promise<void> {
    const checkpoint: Checkpoint = {
      ...state,
      memory: {
        shortTerm: this.shortTermBuffer.map((e) => e.id),
        promotedIds: [],
      },
      createdAt: new Date().toISOString(),
    };

    const checkpointPath = path.join(this.clawvaultPath, "last-checkpoint.json");
    await fs.writeFile(checkpointPath, JSON.stringify(checkpoint, null, 2), "utf-8");
  }

  async loadCheckpoint(): Promise<Checkpoint | null> {
    const checkpointPath = path.join(this.clawvaultPath, "last-checkpoint.json");
    try {
      const raw = await fs.readFile(checkpointPath, "utf-8");
      return JSON.parse(raw) as Checkpoint;
    } catch {
      return null;
    }
  }

  // ── Trade Memory ─────────────────────────────────────────────────────

  async recordTrade(trade: {
    token: string;
    mint: string;
    side: "long" | "short" | "buy" | "sell";
    size: number;
    entryPrice: number;
    exitPrice?: number;
    pnlUsd?: number;
    pnlPct?: number;
    rationale: string;
    signals: Record<string, unknown>;
    outcome?: "win" | "loss" | "neutral";
  }): Promise<VaultEntry> {
    const pnlStr =
      trade.pnlUsd !== undefined
        ? `PnL: $${trade.pnlUsd.toFixed(2)} (${(trade.pnlPct ?? 0).toFixed(2)}%)`
        : "Position open";

    const content = `
## Trade: ${trade.side.toUpperCase()} ${trade.token}

**Token:** ${trade.token} (${trade.mint})
**Side:** ${trade.side}
**Size:** ${trade.size}
**Entry:** $${trade.entryPrice}
**Exit:** ${trade.exitPrice ? `$${trade.exitPrice}` : "Open"}
**${pnlStr}**
**Outcome:** ${trade.outcome ?? "Pending"}

### Rationale
${trade.rationale}

### Signals at Entry
\`\`\`json
${JSON.stringify(trade.signals, null, 2)}
\`\`\`
`.trim();

    return this.remember(content, {
      category: "trades",
      title: `${trade.side} ${trade.token} @ $${trade.entryPrice}`,
      tags: [trade.token, trade.side, trade.outcome ?? "open"],
      metadata: {
        mint: trade.mint,
        entryPrice: trade.entryPrice,
        exitPrice: trade.exitPrice,
        pnlUsd: trade.pnlUsd,
        outcome: trade.outcome,
      },
      score: trade.outcome === "win" ? 0.9 : trade.outcome === "loss" ? 0.7 : 0.5,
    });
  }

  async getTradeHistory(
    token?: string,
    limit = 20
  ): Promise<VaultEntry[]> {
    return this.recall(token ?? "trade", {
      category: "trades",
      limit,
    });
  }

  // ── Reflect (overnight consolidation) ───────────────────────────────

  /**
   * Promote high-value inbox entries to their proper categories.
   * Run this during sleep/checkpoint cycles.
   */
  async reflect(): Promise<{ promoted: number; archived: number }> {
    const inbox = await this.loadAllEntries("inbox");
    let promoted = 0;
    let archived = 0;

    for (const entry of inbox) {
      if (entry.score >= 0.6) {
        // Re-route to proper category
        const newCategory = this.autoRoute(entry.content);
        if (newCategory !== "inbox") {
          await this.remember(entry.content, {
            category: newCategory,
            title: entry.title,
            tags: entry.tags,
            metadata: entry.metadata,
            score: entry.score,
          });

          // Delete from inbox
          const oldPath = this.entryPath(entry);
          await fs.unlink(oldPath).catch(() => null);
          promoted++;
        }
      } else if (entry.score < 0.2) {
        // Archive low-value entries
        const oldPath = this.entryPath(entry);
        await fs.unlink(oldPath).catch(() => null);
        archived++;
      }
    }

    return { promoted, archived };
  }

  // ── Internal Utilities ───────────────────────────────────────────────

  private autoRoute(content: string): VaultCategory {
    const lower = content.toLowerCase();
    const scores: Record<VaultCategory, number> = {
      decisions: 0, lessons: 0, trades: 0, research: 0,
      tasks: 0, backlog: 0, inbox: 0,
    };

    for (const [cat, keywords] of Object.entries(CATEGORY_KEYWORDS) as [VaultCategory, string[]][]) {
      for (const kw of keywords) {
        if (lower.includes(kw)) scores[cat]++;
      }
    }

    const best = Object.entries(scores).sort(([, a], [, b]) => b - a)[0];
    return best[1] > 0 ? (best[0] as VaultCategory) : "inbox";
  }

  private scoreContent(content: string): number {
    const lower = content.toLowerCase();
    let score = 0.3; // base

    // Boost for specific signals
    if (lower.includes("critical") || lower.includes("important")) score += 0.2;
    if (lower.includes("pnl") || lower.includes("profit") || lower.includes("loss")) score += 0.15;
    if (lower.includes("pattern") || lower.includes("insight")) score += 0.15;
    if (lower.includes("learned") || lower.includes("mistake")) score += 0.2;
    if (content.includes("```")) score += 0.1; // code = structured
    if (content.length > 500) score += 0.1;

    return Math.min(score, 1.0);
  }

  private computeRelevance(query: string, entry: VaultEntry): number {
    const queryWords = query.toLowerCase().split(/\s+/);
    const contentWords = new Set(entry.content.toLowerCase().split(/\s+/));
    const titleWords = new Set(entry.title.toLowerCase().split(/\s+/));

    let matches = 0;
    for (const word of queryWords) {
      if (word.length < 3) continue;
      if (contentWords.has(word)) matches += 1;
      if (titleWords.has(word)) matches += 2; // title match weighted more
      if (entry.tags.some((t) => t.toLowerCase().includes(word))) matches += 1.5;
    }

    return Math.min(matches / (queryWords.length * 4), 1.0);
  }

  private extractTitle(content: string): string {
    const headerMatch = content.match(/^#{1,3}\s+(.+)/m);
    if (headerMatch) return headerMatch[1].trim();

    const firstLine = content.split("\n")[0].trim();
    return firstLine.slice(0, 60) + (firstLine.length > 60 ? "..." : "");
  }

  private extractTags(content: string): string[] {
    const tags: string[] = [];
    const tokenMatches = content.match(/\$[A-Z]{2,10}/g) ?? [];
    tags.push(...tokenMatches.map((t) => t.slice(1).toLowerCase()));

    const tagMatches = content.match(/#([a-z][a-z0-9-]*)/gi) ?? [];
    tags.push(...tagMatches.map((t) => t.slice(1).toLowerCase()));

    return [...new Set(tags)].slice(0, 10);
  }

  private generateId(category: VaultCategory): string {
    const date = format(new Date(), "yyyyMMdd");
    const random = Math.random().toString(36).slice(2, 7);
    return `${category}-${date}-${random}`;
  }

  private entryPath(entry: VaultEntry): string {
    const filename = `${entry.id}.md`;
    return path.join(this.vaultPath, entry.category, filename);
  }

  private async loadEntry(filePath: string): Promise<VaultEntry | null> {
    try {
      const raw = await fs.readFile(filePath, "utf-8");
      const parsed = matter(raw);
      const data = parsed.data as Partial<VaultEntry>;

      return {
        id: data.id ?? path.basename(filePath, ".md"),
        category: data.category ?? "inbox",
        title: data.title ?? "",
        content: parsed.content,
        tags: data.tags ?? [],
        links: data.links ?? [],
        score: data.score ?? 0.5,
        createdAt: data.createdAt ?? new Date().toISOString(),
        updatedAt: data.updatedAt ?? new Date().toISOString(),
        metadata: data.metadata ?? {},
      };
    } catch {
      return null;
    }
  }

  private async loadAllEntries(category?: VaultCategory): Promise<VaultEntry[]> {
    const categories: VaultCategory[] = category
      ? [category]
      : ["decisions", "lessons", "trades", "research", "tasks", "backlog", "inbox"];

    const entries: VaultEntry[] = [];
    for (const cat of categories) {
      const dir = path.join(this.vaultPath, cat);
      try {
        const files = await fs.readdir(dir);
        for (const file of files.filter((f) => f.endsWith(".md"))) {
          const entry = await this.loadEntry(path.join(dir, file));
          if (entry) entries.push(entry);
        }
      } catch {
        // Dir may not exist yet
      }
    }

    return entries;
  }

  private async saveGraphIndex(): Promise<void> {
    this.graphIndex.lastUpdated = new Date().toISOString();
    const graphPath = path.join(this.clawvaultPath, "graph-index.json");
    await fs.writeFile(graphPath, JSON.stringify(this.graphIndex, null, 2), "utf-8");
  }

  // ── Context Profile (for agent injection) ───────────────────────────

  async buildContextProfile(query: string): Promise<string> {
    const memories = await this.recall(query, { limit: 5 });
    const recent = this.getShortTermContext(5);
    const trades = await this.getTradeHistory(undefined, 3);

    const sections: string[] = [];

    if (recent.length > 0) {
      sections.push("## Recent Memory (Short-Term)");
      sections.push(recent.map((e) => `- **${e.title}**: ${e.content.slice(0, 100)}...`).join("\n"));
    }

    if (memories.length > 0) {
      sections.push("## Relevant Knowledge");
      sections.push(memories.map((e) => `### ${e.title}\n${e.content.slice(0, 200)}`).join("\n\n"));
    }

    if (trades.length > 0) {
      sections.push("## Recent Trades");
      sections.push(trades.map((t) => `- ${t.title}`).join("\n"));
    }

    return sections.join("\n\n");
  }
}
