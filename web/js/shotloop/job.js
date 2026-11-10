// The Shot Planner's long requests (analysis of a long video, SAM 3, the VLM, combine) run as server jobs: RunPod's
// proxy cuts a request that has had no answer for ~100-120 s, and frontend 1.55.18 aborts api.fetchApi after 60 s.
// The route answers at once with the result, {error}, or {bfs_job: id}; runJob then long-polls /bfs/shotloop/job (the
// server holds each poll at most 25 s) until the job ends, and resolves with exactly what the route used to answer.
// Run the checks: node tests/test_shot_job.mjs

export const JOB_LOST = "ComfyUI restarted (or the job expired) before this finished: click the button again.";

export async function runJob(api, url, init = {}, { onStatus, wait = 20, retryMs = 2000, maxFails = 30, sleep } = {}) {
  const pause = sleep || (ms => new Promise(r => setTimeout(r, ms)));
  const first = await (await api.fetchApi(url, init)).json();
  if (first?.error) throw new Error(first.error);
  if (typeof first?.bfs_job !== "string") return first;
  const poll = `/bfs/shotloop/job?id=${encodeURIComponent(first.bfs_job)}&wait=${wait}`;
  for (let fails = 0; ;) {
    let s;
    try {   // a dropped poll (proxy 502 page, network blip) is asked again: the job keeps running on the server
      const r = await api.fetchApi(poll, { cache: "no-store" });
      s = await r.json();
      if (typeof s?.state !== "string") throw new Error(`HTTP ${r.status}`);
      fails = 0;
    } catch (e) {
      if (++fails > maxFails) {
        throw new Error(`Lost contact with ComfyUI while this was running (${e.message || e}): click the button again to pick it up.`);
      }
      await pause(retryMs);
      continue;
    }
    if (s.state === "done") return s.result;
    if (s.state === "error" || s.state === "lost") throw new Error(s.error || JOB_LOST);
    if (s.status && onStatus) onStatus(s.status);
  }
}
