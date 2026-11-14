// Free-model aggregator: one server, two client wires, ten simulated providers.
//
// The trial hands this server a brief (provider directory, quality table, workload
// manifest) and an env that serves ten free-tier providers on three wires: OpenAI,
// Anthropic and Gemini. The client sends 20 requests, one at a time, on the OpenAI
// or the Anthropic wire, in JSON or streaming, and each must be answered with a
// free model that meets the request's requirements. This server:
//
//   1. reads the brief, probes every provider's catalog route, and builds one
//      catalog document (the same bytes `--list` prints and /register carries);
//   2. listens, registers its port and catalog with the env, and serves the
//      client's GET /v1/models, POST /v1/chat/completions and POST /v1/messages;
//   3. routes each request to the highest-rated free model that meets its
//      requirements, translating the request faithfully into the provider's
//      wire (system text, user text, images, tools, stops, max_tokens and JSON
//      mode all preserved exactly, because the provider echoes what it parsed
//      and the client compares that echo with what it sent);
//   4. on a provider refusal (402, 429, 5xx, unreachable) fails over to the
//      next-best model on another provider, and answers in the wire the client
//      used, with the provider's text whole, its usage counts, and its stop
//      reason mapped to the client's wire;
//   5. polls GET /work and exits 0 when the workload is done.
//
// Everything is a function of the brief, the requests and the providers'
// answers: no wall clock, no unseeded randomness, fixed timestamps, request
// ids derived from arrival order. A replay of the trial sees the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** Fixed timestamps: nothing in any reply reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The register protocol version the env speaks. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** Per-call timeouts, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 10000;
const REGISTER_TIMEOUT_MS = 5000;
/** Largest client request body read, in bytes. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as out for the trial. */
const DEAD_AFTER_REFUSALS = 3;
/** Streaming text is emitted in pieces of this many characters. */
const STREAM_CHUNK = 13;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (e.g. /p/groq) joined to $MAKEFUN_ENV_URL to dial; the catalog's base_url. */
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: ManifestRequest[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` closing the object that opens at `start`, or -1. Strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function directoryProviderOf(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestRequestOf(v: unknown): ManifestRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: Requirement = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  const minContext = asNumber(requires.min_context);
  if (minContext !== null) out.min_context = minContext;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: out,
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = directoryProviderOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingRowOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && asString(v.id) !== '') best.set(asString(v.id), asNumber(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = manifestRequestOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probing: what each provider's own route lists, in its own wire.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. A provider with no key takes no header. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function probedModelOf(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const display =
    asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id;
  return {
    id,
    display_name: display,
    free: asBool(v.free),
    context_length: asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: DirectoryProvider): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [], error: `GET ${url} answered no JSON object` };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [], error: `GET ${url} answered no model list` };
    const models = list.flatMap((v) => {
      const m = probedModelOf(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models, error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

// ---------------------------------------------------------------------------
// Candidates: free models with ratings and capabilities, from the quality
// table first (the sealed truth) and the provider's own listing second.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: DirectoryProvider;
  /** Position of the provider in the brief's directory: the last tie-break. */
  order: number;
  id: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

function ratingRowFor(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one, else the listing's word. */
function isFreeModel(probed: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return probed.free ?? false;
}

/** The routing candidates of one provider: probed models plus any rated model the probe missed. */
function candidatesOf(provider: DirectoryProvider, order: number, probe: Probe, ratings: RatingRow[]): Candidate[] {
  const out = new Map<string, Candidate>();
  for (const model of probe.models) {
    const row = ratingRowFor(ratings, provider.id, model.id);
    if (!isFreeModel(model, row)) continue;
    out.set(model.id, {
      provider,
      order,
      id: model.id,
      rating_milli: row?.rating_milli ?? 0,
      context_length: row?.context_length ?? model.context_length ?? null,
      tools: row?.tools ?? model.tools ?? false,
      vision: row?.vision ?? model.vision ?? false,
      json: row?.json ?? model.json ?? false,
    });
  }
  // A probe that failed (or a listing that hid a model) must not hide a free
  // rated model: the quality table's rows are the sealed truth.
  for (const row of ratings) {
    if (row.provider !== provider.id || !row.free || out.has(row.model)) continue;
    out.set(row.model, {
      provider,
      order,
      id: row.model,
      rating_milli: row.rating_milli,
      context_length: row.context_length,
      tools: row.tools,
      vision: row.vision,
      json: row.json,
    });
  }
  return [...out.values()];
}

function meetsRequirements(requires: Requirement, candidate: Candidate): boolean {
  if (requires.tools === true && !candidate.tools) return false;
  if (requires.vision === true && !candidate.vision) return false;
  if (requires.json === true && !candidate.json) return false;
  if (requires.min_context !== undefined && (candidate.context_length === null || candidate.context_length < requires.min_context)) {
    return false;
  }
  return true;
}

/**
 * The catalog document: every provider in the brief's directory, every model
 * its listing named (or, where the probe failed, every free model the quality
 * table names for it). The base_url is the directory's own path, never joined
 * to the env's origin: the origin differs between a live pass and its replay,
 * and the catalog is part of the state export a replay must reproduce.
 */
function buildCatalog(probes: Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: isFreeModel(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    if (!probe.ok) {
      for (const row of ratings) {
        if (row.provider !== probe.provider.id || !row.free || seen.has(row.model)) continue;
        seen.add(row.model);
        models.push({ id: row.model, free: true, context_length: row.context_length });
      }
      models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// The canonical request: one shape for all three wires, so a translation
// preserves exactly what the provider echoes back (system text, user text,
// images, tool names, stops, max_tokens, JSON mode).
// ---------------------------------------------------------------------------

interface ImagePart {
  /** Media type of a data URL (image/png); null for a plain URL. */
  mediaType: string | null;
  /** Base64 payload of a data URL or base64 source; null for a plain URL. */
  base64: string | null;
  /** Plain URL of an http(s) image; null for a data URL. */
  url: string | null;
}

type ContentPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; image: ImagePart }
  | { kind: 'tool_call'; id: string; name: string; args: string }
  | { kind: 'tool_result'; id: string; text: string };

interface CanonicalMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: ContentPart[];
}

interface CanonicalTool {
  name: string;
  description: string;
  schema: Json;
}

interface CanonicalRequest {
  system: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
  topP: number | undefined;
  toolChoice: 'auto' | 'required' | 'none' | { name: string } | null;
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** Split an image URL into its media type and base64 payload, or keep a plain URL whole. */
function parseImageUrl(url: string): ImagePart {
  const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { mediaType: data[1] !== '' ? data[1] : 'image/png', base64: data[2], url: null };
  return { mediaType: null, base64: null, url };
}

function imageToUrl(image: ImagePart): string {
  if (image.base64 !== null) return `data:${image.mediaType ?? 'image/png'};base64,${image.base64}`;
  return image.url ?? '';
}

function openAIParts(content: unknown): ContentPart[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      parts.push({ kind: 'image', image: parseImageUrl(block.image_url.url) });
    }
  }
  return parts;
}

function anthropicParts(content: unknown): ContentPart[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({
          kind: 'image',
          image: { mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', base64: source.data, url: null },
        });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ kind: 'image', image: { mediaType: null, base64: null, url: source.url } });
      }
    } else if (block.type === 'tool_use') {
      parts.push({
        kind: 'tool_call',
        id: asString(block.id),
        name: asString(block.name),
        args: isRecord(block.input) ? JSON.stringify(block.input) : '{}',
      });
    } else if (block.type === 'tool_result') {
      parts.push({ kind: 'tool_result', id: asString(block.tool_use_id), text: blocksText(block.content) });
    }
  }
  return parts;
}

/** An OpenAI Chat Completions body in canonical form. */
function canonicalFromOpenAI(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const canonical: CanonicalMessage[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (message.role === 'tool') {
      canonical.push({
        role: 'tool',
        parts: [{ kind: 'tool_result', id: asString(message.tool_call_id), text: blocksText(message.content) }],
      });
      continue;
    }
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    const parts = openAIParts(message.content);
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'tool_call',
          id: asString(call.id),
          name: asString(call.function.name),
          args: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
        });
      }
    }
    canonical.push({ role, parts });
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: asString(tool.function.description),
        schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  let toolChoice: CanonicalRequest['toolChoice'] = null;
  if (body.tool_choice === 'auto' || body.tool_choice === 'required' || body.tool_choice === 'none') toolChoice = body.tool_choice;
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { name: body.tool_choice.function.name };
  }
  return {
    system: systemParts.join('\n'),
    messages: canonical,
    tools,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: asNumber(body.temperature) ?? undefined,
    topP: asNumber(body.top_p) ?? undefined,
    toolChoice,
  };
}

/** An Anthropic Messages body in canonical form. */
function canonicalFromAnthropic(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const canonical: CanonicalMessage[] = [];
  for (const message of messages) {
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    canonical.push({ role, parts: anthropicParts(message.content) });
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: asString(tool.description),
        schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonicalRequest['toolChoice'] = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = 'auto';
    else if (body.tool_choice.type === 'any') toolChoice = 'required';
    else if (body.tool_choice.type === 'none') toolChoice = 'none';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { name: body.tool_choice.name };
  }
  return {
    system: blocksText(body.system),
    messages: canonical,
    tools,
    maxTokens: asNumber(body.max_tokens),
    stops: stringList(body.stop_sequences),
    // The Anthropic wire carries no JSON-mode signal: a request on it never
    // echoes json_mode, and the model capability is what the manifest checks.
    jsonMode: false,
    temperature: asNumber(body.temperature) ?? undefined,
    topP: asNumber(body.top_p) ?? undefined,
    toolChoice,
  };
}

/** Canonical parts as OpenAI message content: a string for one text part, else a part array. */
function openAIContent(parts: ContentPart[]): string | Json[] {
  const content: Json[] = [];
  for (const part of parts) {
    if (part.kind === 'text') content.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') content.push({ type: 'image_url', image_url: { url: imageToUrl(part.image) } });
  }
  if (content.length === 1 && parts.length === 1 && parts[0].kind === 'text') return parts[0].text;
  return content;
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
function buildOpenAI(req: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const message of req.messages) {
    if (message.role === 'tool') {
      for (const part of message.parts) {
        if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
      }
      continue;
    }
    const out: Json = { role: message.role, content: openAIContent(message.parts) };
    const calls = message.parts.filter((p): p is Extract<ContentPart, { kind: 'tool_call' }> => p.kind === 'tool_call');
    if (calls.length > 0) {
      out.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
      if (out.content === '') out.content = null;
    }
    messages.push(out);
  }
  const body: Json = { model, messages, stream: false };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.topP !== undefined) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.schema },
    }));
  }
  if (req.toolChoice === 'auto' || req.toolChoice === 'required' || req.toolChoice === 'none') body.tool_choice = req.toolChoice;
  else if (req.toolChoice !== null) body.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

/** Canonical parts as Anthropic message content: a string for one text part, else a block array. */
function anthropicContent(parts: ContentPart[]): string | Json[] {
  const content: Json[] = [];
  for (const part of parts) {
    if (part.kind === 'text') content.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') {
      content.push(
        part.image.base64 !== null
          ? { type: 'image', source: { type: 'base64', media_type: part.image.mediaType ?? 'image/png', data: part.image.base64 } }
          : { type: 'image', source: { type: 'url', url: part.image.url ?? '' } },
      );
    } else if (part.kind === 'tool_call') {
      let input: Json = {};
      try {
        const parsed: unknown = JSON.parse(part.args);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Not JSON: an empty input.
      }
      content.push({ type: 'tool_use', id: part.id, name: part.name, input });
    } else if (part.kind === 'tool_result') {
      content.push({ type: 'tool_result', tool_use_id: part.id, content: part.text });
    }
  }
  if (content.length === 1 && parts.length === 1 && parts[0].kind === 'text') return parts[0].text;
  return content;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
function buildAnthropic(req: CanonicalRequest, model: string): Json {
  const body: Json = { model, stream: false };
  if (req.system !== '') body.system = req.system;
  body.messages = req.messages.map((m) => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: anthropicContent(m.parts) }));
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.topP !== undefined) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (req.toolChoice === 'auto') body.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'required') body.tool_choice = { type: 'any' };
  else if (req.toolChoice === 'none') body.tool_choice = { type: 'none' };
  else if (req.toolChoice !== null) body.tool_choice = { type: 'tool', name: req.toolChoice.name };
  return body;
}

/** The canonical request as a Gemini generateContent body (the model rides in the path). */
function buildGemini(req: CanonicalRequest): Json {
  const body: Json = {};
  if (req.system !== '') body.systemInstruction = { parts: [{ text: req.system }] };
  const contents: Json[] = [];
  for (const message of req.messages) {
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') {
        parts.push(
          part.image.base64 !== null
            ? { inlineData: { mimeType: part.image.mediaType ?? 'image/png', data: part.image.base64 } }
            : { fileData: { fileUri: part.image.url ?? '' } },
        );
      } else if (part.kind === 'tool_call') {
        let args: Json = {};
        try {
          const parsed: unknown = JSON.parse(part.args);
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: empty arguments.
        }
        parts.push({ functionCall: { name: part.name, args } });
      } else if (part.kind === 'tool_result') {
        parts.push({ functionResponse: { name: part.id, response: { output: part.text } } });
      }
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  body.contents = contents;
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })),
      },
    ];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== undefined) config.temperature = req.temperature;
  if (req.topP !== undefined) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Provider calls: one request in the provider's own wire, its answer parsed
// back into text, usage and a stop family. Calls are always non-streaming:
// the answer is buffered and re-emitted in the client's wire, streamed or
// not as the client asked, so a provider's event shape never leaks through.
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
  /** Stop family: 0 end, 1 length, 2 tool call, 3 refusal. */
  family: number;
}

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamilyOf(reason: unknown): number {
  if (typeof reason !== 'string') return 0;
  return STOP_FAMILIES.get(reason) ?? 0;
}

/** The text of an answer's content: a string, or text blocks joined with nothing. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

function parseOpenAIAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.choices) || body.choices.length === 0) return null;
  const choice = body.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = asNumber(usage.prompt_tokens);
  const output = asNumber(usage.completion_tokens);
  if (input === null || output === null) return null;
  return { text: answerText(choice.message.content), input, output, family: stopFamilyOf(choice.finish_reason) };
}

function parseAnthropicAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = asNumber(usage.input_tokens);
  const output = asNumber(usage.output_tokens);
  if (input === null || output === null) return null;
  return { text: answerText(body.content), input, output, family: stopFamilyOf(body.stop_reason) };
}

function parseGeminiAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = body.candidates[0];
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const texts: string[] = [];
  for (const part of candidate.content.parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = asNumber(usage.promptTokenCount);
  const output = asNumber(usage.candidatesTokenCount);
  if (input === null || output === null) return null;
  return { text: texts.join(''), input, output, family: stopFamilyOf(candidate.finishReason) };
}

type CallResult = { ok: true; answer: ProviderAnswer } | { ok: false; status: number; terminal: boolean };

/**
 * Whether a provider's error message says it is done for the trial: its
 * budget spent, or its outage begun. A scheduled refusal ("refused this
 * call") is transient: the provider may answer the next call. The message
 * rides in the wire's own error envelope on all three wires.
 */
function terminalRefusal(text: string): boolean {
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return false;
  }
  const message = isRecord(body) && isRecord(body.error) ? body.error.message : null;
  if (typeof message !== 'string') return false;
  return message.includes('budget is spent') || message.includes(' is out');
}

/** One inference call to a provider. Any refusal, error or unparseable answer is a failure. */
async function callProvider(candidate: Candidate, req: CanonicalRequest): Promise<CallResult> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = buildOpenAI(req, candidate.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = buildAnthropic(req, candidate.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.id)}:generateContent`;
    payload = buildGemini(req);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, terminal: false };
  }
  if (!res.ok) {
    let text = '';
    try {
      text = await res.text();
    } catch {
      // The body is unreadable: the status is what matters.
    }
    return { ok: false, status: res.status, terminal: terminalRefusal(text) };
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return { ok: false, status: 502, terminal: false };
  }
  const answer =
    provider.wire === 'openai' ? parseOpenAIAnswer(body) : provider.wire === 'anthropic' ? parseAnthropicAnswer(body) : parseGeminiAnswer(body);
  if (answer === null) return { ok: false, status: 502, terminal: false };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Responses in the client's wire, with the provider's text whole, its usage
// counts, and its stop reason mapped. Usage is always reported, in every
// shape: a stream without a final usage chunk would fail the usage check.
// ---------------------------------------------------------------------------

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_CHUNK) pieces.push(text.slice(at, at + STREAM_CHUNK));
  return pieces.length === 0 ? [''] : pieces;
}

function openAIStop(family: number): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

function anthropicStop(family: number): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

function openAICompletion(answer: ProviderAnswer, model: string, index: number): Json {
  return {
    id: `chatcmpl-agg-${index}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStop(answer.family),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

function sseData(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function openAIStreamText(answer: ProviderAnswer, model: string, index: number): string {
  const id = `chatcmpl-agg-${index}`;
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseData(chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseData(chunk({ content: piece }, null)));
  frames.push(sseData(chunk({}, openAIStop(answer.family))));
  frames.push(
    sseData({
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicMessage(answer: ProviderAnswer, model: string, index: number): Json {
  return {
    id: `msg_agg-${index}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.family),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

function anthropicStreamText(answer: ProviderAnswer, model: string, index: number): string {
  const id = `msg_agg-${index}`;
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.family), stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

interface ServedModel {
  id: string;
  provider: string;
}

/** GET /v1/models in the OpenAI list shape. */
function openAIModelList(models: ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider, free: true })),
  };
}

/** GET /v1/models in the Anthropic list shape. */
function anthropicModelList(models: ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT, free: true })),
    has_more: false,
  };
}

function openAIError(message: string, type: string): Json {
  return { error: { message, type, param: null, code: null } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Routing with failover, the HTTP server, register, and the work poll.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let candidates: Candidate[] = [];
/** Inference requests served so far: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Provider call counts, for the load-balancing tie-break. */
const attemptsByProvider = new Map<string, number>();
/** Consecutive refusals per provider; DEAD_AFTER_REFUSALS in a row marks it out. */
const refusalsByProvider = new Map<string, number>();
const deadProviders = new Set<string>();

/** Probe every provider and build the one catalog and the candidate list. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.providers) {
    const probe = await probeProvider(ENV_URL, provider);
    if (!probe.ok && probe.error !== null) log(probe.error);
    probes.push(probe);
  }
  catalog = buildCatalog(probes, brief.ratings);
  candidates = [];
  probes.forEach((probe, order) => {
    candidates.push(...candidatesOf(probe.provider, order, probe, brief.ratings));
  });
}

/** The free model ids the catalog registers, de-duplicated, in catalog order. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

/** Whether any message of the canonical request carries an image. */
function hasImage(req: CanonicalRequest): boolean {
  return req.messages.some((m) => m.parts.some((p) => p.kind === 'image'));
}

/** Rank the candidates for one request: rating first, then the least-used provider. */
function rankCandidates(requires: Requirement, excludeAnthropicWire: boolean): Candidate[] {
  const ranked = candidates.filter((c) => {
    if (deadProviders.has(c.provider.id)) return false;
    if (excludeAnthropicWire && c.provider.wire === 'anthropic') return false;
    return meetsRequirements(requires, c);
  });
  ranked.sort((a, b) => {
    if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
    const aAttempts = attemptsByProvider.get(a.provider.id) ?? 0;
    const bAttempts = attemptsByProvider.get(b.provider.id) ?? 0;
    if (aAttempts !== bAttempts) return aAttempts - bAttempts;
    if (a.order !== b.order) return a.order - b.order;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return ranked;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one inference request: the manifest's requirements by arrival order
 * (unioned with what the body itself carries), the best free model that meets
 * them, failover to the next provider on a refusal, and the answer in the
 * client's wire. A provider is tried at most once per request: a refusal may
 * be its budget spent or its outage begun, and the next provider is a call
 * away. A provider whose error says its budget is spent or it is out, or
 * that refuses three times running, is out for the rest of the trial.
 */
async function serveInference(wire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const ordinal = manifestIndex;
  manifestIndex += 1;
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const canonical = wire === 'openai' ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
  const manifestRequires = brief.manifest[ordinal]?.requires ?? {};
  const requires: Requirement = { ...manifestRequires };
  if (canonical.tools.length > 0) requires.tools = true;
  if (hasImage(canonical)) requires.vision = true;
  // A JSON-mode request on the OpenAI wire must reach a wire that can carry
  // the signal (OpenAI's response_format, Gemini's responseMimeType): an
  // Anthropic-wire provider would echo json_mode absent and lose the request.
  const excludeAnthropicWire = canonical.jsonMode;
  const stream = body.stream === true;
  const tried = new Set<string>();
  for (;;) {
    const ranked = rankCandidates(requires, excludeAnthropicWire).filter((c) => !tried.has(c.provider.id));
    if (ranked.length === 0) {
      log(`request ${ordinal + 1}: no provider left to try`);
      sendError(res, wire, 502, 'api_error', 'every free model that meets this request refused it');
      return;
    }
    const candidate = ranked[0];
    tried.add(candidate.provider.id);
    attemptsByProvider.set(candidate.provider.id, (attemptsByProvider.get(candidate.provider.id) ?? 0) + 1);
    const result = await callProvider(candidate, canonical);
    if (result.ok) {
      refusalsByProvider.set(candidate.provider.id, 0);
      const serving = `${candidate.provider.id}/${candidate.id}`;
      log(`request ${ordinal + 1}: served by ${serving}`);
      if (wire === 'openai') {
        if (stream) sendStream(res, openAIStreamText(result.answer, candidate.id, ordinal + 1));
        else sendJson(res, 200, openAICompletion(result.answer, candidate.id, ordinal + 1));
      } else {
        if (stream) sendStream(res, anthropicStreamText(result.answer, candidate.id, ordinal + 1));
        else sendJson(res, 200, anthropicMessage(result.answer, candidate.id, ordinal + 1));
      }
      return;
    }
    const refusals = (refusalsByProvider.get(candidate.provider.id) ?? 0) + 1;
    refusalsByProvider.set(candidate.provider.id, refusals);
    log(`request ${ordinal + 1}: ${candidate.provider.id}/${candidate.id} refused (HTTP ${result.status}); failing over`);
    if (result.terminal) {
      deadProviders.add(candidate.provider.id);
      log(`${candidate.provider.id} says its budget is spent or it is out; treating it as out`);
    } else if (refusals >= DEAD_AFTER_REFUSALS) {
      deadProviders.add(candidate.provider.id);
      log(`${candidate.provider.id} refused ${refusals} times running; treating it as out`);
    }
  }
}

/** Inference requests run one at a time, in arrival order: the manifest's order. */
let inferenceQueue: Promise<void> = Promise.resolve();

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const run = inferenceQueue.then(() => serveInference('openai', req, res));
    inferenceQueue = run.catch(() => undefined);
    await run;
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const run = inferenceQueue.then(() => serveInference('anthropic', req, res));
    inferenceQueue = run.catch(() => undefined);
    await run;
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  const payload = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog });
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: payload,
        signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
