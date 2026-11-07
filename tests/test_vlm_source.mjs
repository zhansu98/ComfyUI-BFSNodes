// The Shot Planner panel's vlmSourceOf (web/js/bfs_shot_loop.js): which loader the VLM buttons ask the server to load.
// Run: node tests/test_vlm_source.mjs   (no browser; the function is lifted out of the panel file with a mock app)
import fs from "node:fs";
import { fileURLToPath } from "node:url";
const src = fs.readFileSync(process.argv[2] ?? fileURLToPath(new URL("../web/js/bfs_shot_loop.js", import.meta.url)), "utf8");
const a = src.indexOf("async function vlmSourceOf"), b = src.indexOf("const DEFAULT_W = 1180");
let calls = 0;
const app = { graph: null, rootGraph: null, graphToPrompt: async () => { calls++; return { workflow: {}, output: app._out }; } };
const vlmSourceOf = new Function("app", src.slice(a, b) + "; return vlmSourceOf;")(app);
const mkGraph = (nodes, links) => ({ getNodeById: id => nodes.find(n => n.id === id) ?? null, links: new Map(links.map(l => [l.id, l])) });
const loader = (id, extra = {}) => ({ id, type: "CLIPLoader", comfyClass: "CLIPLoader", mode: 0,
  widgets: [{ name: "clip_name", value: "qwen3vl_8b_bf16.safetensors" }, { name: "type", value: "qwen_image" }, { name: "device", value: "default" }], inputs: [], ...extra });
const planner = (id, link) => ({ id, type: "BFSShotPlanner", inputs: [{ name: "vlm", link }] });
const check = (name, got, want) => { const ok = JSON.stringify(got) === JSON.stringify(want); console.log(ok ? "ok  " : "FAIL", name, ok ? "" : JSON.stringify(got)); if (!ok) process.exitCode = 1; };
// 1 direct wire in root graph
let L = loader(5), P = planner(7, 1), g = mkGraph([L, P], [{ id: 1, origin_id: 5, origin_slot: 0 }]); P.graph = g; app.graph = app.rootGraph = g;
check("direct root", await vlmSourceOf(P), { class_type: "CLIPLoader", inputs: { clip_name: "qwen3vl_8b_bf16.safetensors", type: "qwen_image", device: "default" }, slot: 0 });
check("no graphToPrompt for direct", calls, 0);
// 2 planner in a subgraph with id 7 wired to its own loader, root also has planner 7 with another loader
const L2 = loader(5, { widgets: [{ name: "clip_name", value: "qwen3vl_4b.safetensors" }] }); const P2 = planner(7, 1);
const sg = mkGraph([L2, P2], [{ id: 1, origin_id: 5, origin_slot: 0 }]); P2.graph = sg;
check("subgraph planner gets its own loader", (await vlmSourceOf(P2)).inputs.clip_name, "qwen3vl_4b.safetensors");
// 3 muted / bypassed loader -> null; unconnected -> null
L.mode = 2; check("muted loader", await vlmSourceOf(P), null); L.mode = 4; check("bypassed loader", await vlmSourceOf(P), null); L.mode = 0;
check("unconnected", await vlmSourceOf(planner(8, null)), null);
// 4 linked input marked as link
L.inputs = [{ name: "clip_name", widget: { name: "clip_name" }, link: 9 }]; g.links.set(9, { id: 9, origin_id: 3, origin_slot: 0 });
check("linked input", (await vlmSourceOf(P)).inputs.clip_name, ["3", 0]); L.inputs = [];
// 5 virtual hop in root -> graphToPrompt, exact id only
const V = { id: 6, isVirtualNode: true }; const P3 = planner(7, 2); const g3 = mkGraph([V, P3], [{ id: 2, origin_id: 6, origin_slot: 0 }]); P3.graph = g3; app.graph = app.rootGraph = g3;
app._out = { "7": { class_type: "BFSShotPlanner", inputs: { vlm: ["5", 0] } }, "5": { class_type: "CLIPLoader", inputs: { clip_name: "x" } } };
check("virtual hop via graphToPrompt", await vlmSourceOf(P3), { class_type: "CLIPLoader", inputs: { clip_name: "x" }, slot: 0 });
app._out = { "100:7": { class_type: "BFSShotPlanner", inputs: { vlm: ["5", 0] } } };
check("no unrelated fallback", await vlmSourceOf(P3), null);
// 6 virtual hop inside a subgraph -> throws; graphToPrompt error propagates
const P4 = planner(7, 2); P4.graph = mkGraph([V, P4], [{ id: 2, origin_id: 6, origin_slot: 0 }]);
check("subgraph virtual hop throws", await vlmSourceOf(P4).then(() => "no throw", e => /subgraph/.test(e.message)), true);
app.graphToPrompt = async () => { throw new Error("webcam"); };
check("serialization error propagates", await vlmSourceOf(P3).then(() => "no throw", e => e.message), "webcam");
// 7 a widget's serializeValue is what is sent (as a queue does), not its displayed value
app.graphToPrompt = async () => { throw new Error("must not serialize the graph"); };
const L7 = loader(5, { widgets: [{ name: "clip_name", value: "Qwen 8B (display)", serializeValue: async (n, i) => `real_${i}.safetensors` }] });
const P7 = planner(7, 1); P7.graph = mkGraph([L7, P7], [{ id: 1, origin_id: 5, origin_slot: 0 }]); app.graph = app.rootGraph = P7.graph;
check("serializeValue used", (await vlmSourceOf(P7)).inputs.clip_name, "real_0.safetensors");
if (process.exitCode) console.log("FAILED"); else console.log("all ok");
