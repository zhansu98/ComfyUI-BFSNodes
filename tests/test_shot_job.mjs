// The Shot Planner's job client (web/js/shotloop/job.js runJob): a long request answered with a job id is followed
// with short long-polls until it ends. Run: node tests/test_shot_job.mjs   (no browser; mock api)
import { runJob, JOB_LOST } from "../web/js/shotloop/job.js";

const check = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(ok ? "ok  " : "FAIL", name, ok ? "" : `got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
  if (!ok) process.exitCode = 1;
};
const res = (body, status = 200) => ({ status, json: async () => (body instanceof Error ? (() => { throw body; })() : body) });
// replies: a list of what each fetchApi call answers (a value, or a function of (url, init)); calls are recorded
const mockApi = replies => {
  const calls = [];
  return { calls, fetchApi: async (url, init) => { calls.push({ url, init }); const r = replies.shift(); if (r instanceof Error) throw r; return typeof r === "function" ? r(url, init) : r; } };
};
const noSleep = { sleep: async () => {} };
const fails = p => p.then(() => "no throw", e => e.message);

// 1 the route answered with the result: returned as is, no poll
let api = mockApi([res({ segs: [1] })]);
check("inline result", await runJob(api, "/bfs/shotloop/plan", { method: "POST" }, noSleep), { segs: [1] });
check("no poll for inline", api.calls.length, 1);

// 2 the route answered {error}: thrown with that message (the panel shows it as before)
api = mockApi([res({ error: "ValueError: nope" }, 500)]);
check("inline error", await fails(runJob(api, "/bfs/shotloop/vlm", {}, noSleep)), "ValueError: nope");

// 3 job -> running (progress reaches onStatus) -> done
const seen = [];
api = mockApi([res({ bfs_job: "abc" }), res({ state: "running", status: { stage: "Reading the video", done: 3, total: 10 } }),
               res({ state: "done", result: { n: 42 } })]);
check("job result", await runJob(api, "/bfs/shotloop/analyze?video=a.mp4&fps=24", undefined, { ...noSleep, onStatus: s => seen.push(s) }), { n: 42 });
check("progress passed on", seen, [{ stage: "Reading the video", done: 3, total: 10 }]);
check("poll url", api.calls[1].url, "/bfs/shotloop/job?id=abc&wait=20");
check("poll not cached", api.calls[1].init, { cache: "no-store" });

// 4 the job failed / ComfyUI restarted: thrown with the server's message
api = mockApi([res({ bfs_job: "e" }), res({ state: "error", error: "RuntimeError: boom" })]);
check("job error", await fails(runJob(api, "/x", {}, noSleep)), "RuntimeError: boom");
api = mockApi([res({ bfs_job: "l" }), res({ state: "lost", error: JOB_LOST })]);
check("job lost", await fails(runJob(api, "/x", {}, noSleep)), JOB_LOST);

// 5 a dropped poll (network error, proxy HTML page) is asked again; the job still resolves
api = mockApi([res({ bfs_job: "r" }), new TypeError("Failed to fetch"), res(new SyntaxError("Unexpected token <"), 502),
               res({ state: "done", result: "ok" })]);
check("poll retried", await runJob(api, "/x", {}, noSleep), "ok");

// 6 too many dropped polls in a row: a clear message
api = mockApi([res({ bfs_job: "m" }), new TypeError("Failed to fetch"), new TypeError("Failed to fetch"), new TypeError("Failed to fetch")]);
check("lost contact", /Lost contact with ComfyUI/.test(await fails(runJob(api, "/x", {}, { ...noSleep, maxFails: 2 }))), true);

// 7 the requests carry no frontend timeout override (each one is short now) and the route's init is passed through
api = mockApi([res({ bfs_job: "t" }), res({ state: "done", result: 1 })]);
await runJob(api, "/bfs/shotloop/mask", { method: "POST", body: "{}" }, noSleep);
check("start init passed through", api.calls[0].init, { method: "POST", body: "{}" });
check("no timeoutMs", api.calls.some(c => c.init && "timeoutMs" in c.init), false);

if (!process.exitCode) console.log("all ok");
