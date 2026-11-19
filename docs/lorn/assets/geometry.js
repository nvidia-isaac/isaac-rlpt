(() => {
/* ---------- utilities ---------- */
const root = document.querySelector('.viz-root');
const css = n => getComputedStyle(root).getPropertyValue(n).trim();
let C = {}, DARK = false;
const RAMP_L = ['#e9e5f3', '#d4cce8', '#bdb0d8', '#a295c7', '#8879b3', '#70619b', '#55467d'];
let RAMP = RAMP_L, RAMP_RGB = [];
const hex2rgb = h => [0, 2, 4].map(i => parseInt(h.replace('#', '').substr(i, 2), 16));
function readColors() {
  C = { r: css('--series-3'), ink: css('--text-primary'), ink2: css('--text-secondary'), muted: css('--muted'), grid: css('--grid'), axis: css('--axis'), surf: css('--surface-1'), page: css('--page') };
  const pg = hex2rgb(C.page); DARK = (pg[0] + pg[1] + pg[2]) / 3 < 100;
  RAMP = DARK ? RAMP_L.slice().reverse() : RAMP_L; RAMP_RGB = RAMP.map(hex2rgb);
}
readColors();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); updateLegend(); invalidate(); });
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
let STATIC = reduced, MODE = 'adv', HEAT = 'est';
function heatRGB(v) { v = Math.max(0, Math.min(1, v)) * (RAMP_RGB.length - 1); const i = Math.min(RAMP_RGB.length - 2, Math.floor(v)), f = v - i, a = RAMP_RGB[i], b = RAMP_RGB[i + 1]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
const DIV_NEG = [207, 153, 103], DIV_POS = [111, 101, 175];
function divRGB(x) { x = Math.max(-1, Math.min(1, x)); const mid = DARK ? [56, 56, 53] : [243, 241, 248], e = x < 0 ? DIV_NEG : (DARK ? [57, 135, 229] : DIV_POS), f = Math.abs(x); return [mid[0] + (e[0] - mid[0]) * f, mid[1] + (e[1] - mid[1]) * f, mid[2] + (e[2] - mid[2]) * f]; }
const valRGB = (q, nat) => MODE === 'adv' ? divRGB((q - nat) / M.AMAX) : heatRGB((q - M.ABSLO) / (M.ABSHI - M.ABSLO));
const heat = v => { const c = heatRGB(v); return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; };
const valCol = (q, nat) => { const c = valRGB(q, nat); return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; };
const hexA = (hex, a) => { const c = hex2rgb(hex); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
function fit(cv) { const dpr = devicePixelRatio || 1, w = cv.clientWidth; if (!cv.dataset.h) cv.dataset.h = cv.getAttribute('height'); const h = +cv.dataset.h;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px'; }
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h); return { ctx, w, h, dpr }; }
const ease = p => p * p * (3 - 2 * p), clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = v => { const l = Math.hypot(...v); return v.map(x => x / l); };
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ---------- toy model: banded values at every decision (seed 21) ---------- */
const sig = x => 1 / (1 + Math.exp(-x));
const M = (() => {
  const rng = mulberry(15), g = () => { let u = 0; while (!u) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); };
  const randOrth = gg => { const a = nrm([gg(), gg(), gg()]); let b = [gg(), gg(), gg()]; const d = dot3(a, b); return [a, nrm(b.map((x, i) => x - d * a[i]))]; };
  const [p1, p2] = randOrth(g);
  const T = 32, K = 6, NT = 96, L0 = -1.6, cA = 0.16;
  const rot = (v, ax, a) => { const c = Math.cos(a), s = Math.sin(a), k = nrm(ax), kv = cross(k, v), kd = dot3(k, v); return v.map((x, i) => x * c + kv[i] * s + k[i] * kd * (1 - c)); };
  // ten islands of high or low value: drifting centres, varied widths, weights that swell and fade, plus a faint band
  const r3 = () => nrm([g(), g(), g()]);
  const B = [...Array(10)].map((_, k) => ({ c: r3(), ax: r3(), w: (0.05 + 0.12 * rng()) * (rng() < 0.5 ? -1 : 1), wob: 0.4 * rng(), wf: 0.15 + 0.25 * rng(), kap: 3 + 11 * rng(), A: (k % 3 === 2 ? -1 : 1) * (0.6 + 0.7 * rng()), nu: 0.12 + 0.25 * rng(), ps: 6.28 * rng() }));
  const band = { c: r3(), ax: r3(), w: 0.04 };
  const CEN = [...Array(T)].map((_, t) => B.map(b => rot(b.c, b.ax, b.w * t + b.wob * Math.sin(b.wf * t))));
  const WGT = [...Array(T)].map((_, t) => B.map(b => b.A * Math.max(0, 0.25 + 0.75 * Math.sin(b.nu * t + b.ps))));
  const BJ = [...Array(T)].map((_, t) => rot(band.c, band.ax, band.w * t));
  const AT = [...Array(T)].map((_, t) => cA * (0.75 + 0.7 * Math.exp(-((t - 7) ** 2) / 3) + 0.45 * Math.exp(-((t - 18) ** 2) / 4) + 0.8 * Math.exp(-((t - 26) ** 2) / 2.5)));
  const h = (t, x) => { let s = 0; for (let k = 0; k < B.length; k++) { const w = WGT[t][k]; if (w) s += w * Math.exp(B[k].kap * (dot3(x, CEN[t][k]) - 1)); } return s + 0.18 * Math.sin(Math.PI * 1.6 * dot3(x, BJ[t])); };
  const F = 1500, fib = []; for (let i = 0; i < F; i++) { const y = 1 - 2 * (i + 0.5) / F, r = Math.sqrt(1 - y * y), ph = i * Math.PI * (3 - Math.sqrt(5)); fib.push([r * Math.cos(ph), y, r * Math.sin(ph)]); }
  const th = [...Array(NT)].map((_, i) => -Math.PI + 2 * Math.PI * i / NT);
  const circOf = (q1, q2) => a => q1.map((x, i) => Math.cos(a) * x + Math.sin(a) * q2[i]);
  const circ = circOf(p1, p2), H = [...Array(T)].map((_, t) => th.map(a => h(t, circ(a))));
  const Hs = [...Array(T)].map((_, t) => fib.map(x => h(t, x)));
  const [q1, q2] = randOrth((() => { const r2 = mulberry(123); return () => { let u = 0; while (!u) u = r2(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r2()); }; })());
  const circ2 = circOf(q1, q2), H2 = [...Array(T)].map((_, t) => th.map(a => h(t, circ2(a))));
  const hAt = (Hm, t, a) => { let f = ((a + Math.PI) / (2 * Math.PI)) * NT; f = ((f % NT) + NT) % NT; const i = Math.floor(f), j = (i + 1) % NT, w = f - i; return Hm[t][i] * (1 - w) + Hm[t][j] * w; };
  const mrng = mulberry(5); const mc = pick => { let s = 0; for (let n = 0; n < 12000; n++) { let L = L0; for (let t = 0; t < T; t++) L += AT[t] * pick(t); s += sig(L); } return s / 12000; };
  const native = mc(t => Hs[t][Math.floor(mrng() * F)]), uni = mc(t => H[t][Math.floor(mrng() * NT)]);
  const logit = (Hm, p) => L0 + p.reduce((a, x, t) => a + AT[t] * hAt(Hm, t, x), 0);
  const base = (Hm, p) => p.map((x, t) => logit(Hm, p) - AT[t] * hAt(Hm, t, x));
  let best = null; for (let i = 0; i < NT; i++) { const p = [...Array(T)].map((_, t) => th[i] + 0.25 * Math.sin(t / 4)); const s = sig(logit(H, p)); if (!best || Math.abs(s - uni) < best.d) best = { p, d: Math.abs(s - uni) }; }
  // RL from rollouts: the actor never sees the true map, only a critic estimate fitted to the rollouts so far.
  // Each rollout visits slightly different states (jitter), samples around the current path, and updates the estimate there.
  const qmap = p => { const bs = base(H, p), q = H.map((row, t) => row.map(hv => sig(bs[t] + AT[t] * hv))), nat = Hs.map((row, t) => row.reduce((a, hv) => a + sig(bs[t] + AT[t] * hv), 0) / F); return { q, nat }; };
  const er = mulberry(777), eg = () => { let u = 0; while (!u) u = er(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * er()); };
  const EX = [...Array(T)].map(() => [...Array(6)].map(() => eg()));
  const jr = mulberry(31), jg = () => { let u = 0; while (!u) u = jr(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * jr()); };
  const sigmaOf = k => 0.12 + 0.95 * (1 - k / K), dth = 2 * Math.PI / NT;
  const est = [...Array(T)].map(() => new Float64Array(NT)), cnt = [...Array(T)].map(() => new Float64Array(NT));
  let path = best.p.map(wrap); const paths = [path.slice()], EST = [], ROLL = [];
  for (let k = 0; k < K; k++) {
    const { q, nat } = qmap(path), ph = 6.28 * jr();
    const shift = [...Array(T)].map((_, t) => Math.round((0.22 * Math.sin(0.3 * t + ph) + 0.08 * jg()) / dth)); // this rollout's states
    const samples = []; path.forEach((a, t) => { samples.push([t, a]); EX[t].forEach(e => samples.push([t, wrap(a + sigmaOf(k) * e)])); });
    samples.forEach(([ts, as]) => { for (let t = Math.max(0, ts - 2); t <= Math.min(T - 1, ts + 2); t++) for (let i = 0; i < NT; i++) {
      const kern = Math.exp(-(wrap(th[i] - as) ** 2) / (2 * 0.3 * 0.3)) * Math.exp(-((t - ts) ** 2) / 2); if (kern < 0.02) continue;
      const obs = q[t][(i + shift[t] + NT) % NT] - nat[t]; est[t][i] = (cnt[t][i] * est[t][i] + kern * obs) / (cnt[t][i] + kern); cnt[t][i] += kern; } });
    EST.push(est.map((row, t) => Array.from(row, (v, i) => v * (1 - Math.exp(-cnt[t][i]))))); ROLL.push(shift);
    const tgt = path.map((x, t) => { let bi = -1, bv = 0; for (let i = 0; i < NT; i++) { const v = EST[k][t][i]; if (v > bv) { bv = v; bi = i; } } return bi < 0 ? x : th[bi]; });
    let np = path.map((x, t) => wrap(x + 0.6 * wrap(tgt[t] - x)));
    np = np.map((x, t) => { const a = np[Math.max(0, t - 1)], c = np[Math.min(T - 1, t + 1)]; return wrap(x + 0.5 * (0.5 * wrap(a - x) + 0.5 * wrap(c - x))); });
    path = np; paths.push(path.slice()); }
  // Q_t^pi(hval): value of using a tensor with score hval at decision t, all other decisions following the path
  const BASE = paths.map(p => base(H, p)), succ = paths.map(p => sig(logit(H, p)));
  const qOf = (k, t, hv) => sig(BASE[k][t] + AT[t] * hv);
  const Q = paths.map((p, k) => H.map((row, t) => row.map(hv => qOf(k, t, hv))));
  const NAT = paths.map((p, k) => Hs.map((row, t) => row.reduce((a, hv) => a + qOf(k, t, hv), 0) / F));
  const BASEs = base(H2, paths[K]), Qswap = H2.map((row, t) => row.map(hv => sig(BASEs[t] + AT[t] * hv))), NATswap = Hs.map((row, t) => row.reduce((a, hv) => a + sig(BASEs[t] + AT[t] * hv), 0) / F);
  const succSwap = sig(logit(H2, paths[K]));
  // shared scale for the advantage view: 97th percentile of |Q - native| on the subspace
  const devs = []; [0, K].forEach(k => Q[k].forEach((row, t) => row.forEach(v => devs.push(Math.abs(v - NAT[k][t]))))); devs.sort((a, b) => a - b);
  const AMAX = devs[Math.floor(devs.length * 0.97)];
  const strong = AT.map(a => a > cA * 1.3);
  let ABSLO = 1, ABSHI = 0; Q.forEach(qk => qk.forEach(row => row.forEach(v => { ABSLO = Math.min(ABSLO, v); ABSHI = Math.max(ABSHI, v); }))); Qswap.forEach(row => row.forEach(v => { ABSLO = Math.min(ABSLO, v); ABSHI = Math.max(ABSHI, v); }));
  const nn = cross(p1, p2), be = 68 * Math.PI / 180, VZ = nrm(nn.map((x, i) => Math.cos(be) * x + Math.sin(be) * p2[i])), VX = p1, VY = cross(VX, VZ);
  const toWorld = (x, y, z) => [0, 1, 2].map(i => x * VX[i] + y * VY[i] + z * VZ[i]), toView = c => [dot3(c, VX), dot3(c, VY), dot3(c, VZ)];
  const cm = paths[K].reduce((a, x) => [a[0] + Math.cos(x), a[1] + Math.sin(x)], [0, 0]), phiStar = -Math.atan2(cm[1], cm[0]);
  return { T, K, NT, th, p1, p2, h, circ, toWorld, toView, phiStar, ABSLO, ABSHI, EST, EX, sigmaOf, Q, NAT, Qswap, NATswap, paths, succ, succSwap, native, uni, qOf, AMAX, strong };
})();
document.getElementById('roNat').textContent = (M.native * 100).toFixed(0) + '%';
document.getElementById('roUni').textContent = (M.uni * 100).toFixed(0) + '%';

/* ---------- sphere rendering (cached per decision) ---------- */
function sphereImage(t, k, Rpx) { // view coords = world coords; front hemisphere z > 0
  const size = Math.ceil(2 * Rpx) + 2, img = new ImageData(size, size), d = img.data, nat = M.NAT[k][t];
  for (let py = 0; py < size; py++) for (let px = 0; px < size; px++) {
    const x = (px + 0.5 - size / 2) / Rpx, y = -(py + 0.5 - size / 2) / Rpx, rr = x * x + y * y; if (rr > 1) continue;
    const z = Math.sqrt(1 - rr), c = valRGB(M.qOf(k, t, M.h(t, M.toWorld(x, y, z))), nat), shade = 0.78 + 0.22 * z, o = (py * size + px) * 4;
    d[o] = c[0] * shade; d[o + 1] = c[1] * shade; d[o + 2] = c[2] * shade; d[o + 3] = rr > 0.97 ? 255 * (1 - (rr - 0.97) / 0.03) : 255; }
  const cv = document.createElement('canvas'); cv.width = size; cv.height = size; cv.getContext('2d').putImageData(img, 0, 0); return cv;
}
let sphCache = null, sphCacheKey = '';
function invalidate() { sphCache = null; sphCacheKey = ''; drawSph(); drawRing(); drawIter(); }

/* ---------- Figure 1: spheres → tube → RL on the tube ---------- */
const DUR = [3.5, 3.4, 3.0, 3.2, 11.4], START = DUR.reduce((a, d, i) => (a.push(i ? a[i - 1] + DUR[i - 1] : 0), a), []), TOTAL = DUR.reduce((a, b) => a + b, 0);
const TITLES = ['Each observation has its own landscape of useful noise inputs (geometric illustration)',
  'One fixed subspace intersects every noise sphere in the same coordinates',
  'Across an episode, these shared coordinates form a tube',
  'Unroll the tube: decisions across, noise choices up and down',
  'The actor learns from rollouts, guided by the critic’s estimated values'];
const STAGE_NAMES = ['value spheres', 'cut the subspaces', 'stack into a tube', 'unroll', 'RL on the strip'];
let tau = 0, playing = !STATIC, staticStage = 4;
const playBtn = document.getElementById('playBtn');
playBtn.textContent = playing ? 'Pause' : 'Play';
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; };
document.querySelectorAll('#stageSeg button').forEach(b => b.onclick = () => { const st = +b.dataset.st; staticStage = st; tau = playing ? START[st] + 0.01 : START[st] + DUR[st] - 0.01; });
document.querySelectorAll('.bar .seg button').forEach(b => b.addEventListener('click', () => { document.querySelectorAll('.bar .seg button').forEach(x => x.setAttribute('aria-pressed', x === b)); STATIC = b.dataset.mode === 'static';
  if (STATIC) { playing = false; tau = START[staticStage] + DUR[staticStage] - 0.01; } else { playing = true; } playBtn.textContent = playing ? 'Pause' : 'Play'; invalidate(); }));
if (STATIC) { document.querySelectorAll('.bar .seg button').forEach(x => x.setAttribute('aria-pressed', x.dataset.mode === 'static')); tau = START[4] + DUR[4] - 0.01; }
// exploration offsets (fixed, so the cloud is stable from frame to frame)
const EXPL_UNUSED = (() => { const r = mulberry(777), g = () => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }; return [...Array(M.T)].map(() => [...Array(6)].map(() => g())); })();
const angLerp = (a, b, f) => wrap(a + f * wrap(b - a));
function pathAt(k, f) { // path between iterations k and k+1
  const P0 = M.paths[k], P1 = M.paths[Math.min(M.K, k + 1)]; return P0.map((a, t) => angLerp(a, P1[t], f)); }

function drawTube(dt) {
  if (playing && !window.sitePaused) tau = (tau + dt) % TOTAL;
  let st = 0; while (st < 4 && tau >= START[st + 1]) st++;
  const p = clamp((tau - START[st]) / DUR[st]);
  document.querySelectorAll('#stageSeg button').forEach(b => b.setAttribute('aria-pressed', +b.dataset.st === st));
  const cv = document.getElementById('tube'), { ctx, w, h, dpr } = fit(cv);
  const T = M.T, NT = M.NT, top = 34;
  // sphere grid
  const cols = 8, rows = 4, cw = (w - 20) / cols, chh = (h - top - 16) / rows, Rs = Math.min(cw, chh) * 0.38;
  const sc = t => [10 + cw * (t % cols + 0.5), top + chh * (Math.floor(t / cols) + 0.5) - 4];
  // tunnel geometry: decision 1 is near and large, decision 32 far and small
  const xA = 60, xB = w - 40, cy0 = top + (h - top) / 2 + 12, R0 = Math.min(150, (h - top) * 0.36);
  const sOf = u => 1 / (1 + 1.6 * u), sFar = sOf(1);
  const ax = t => { const u = t / (T - 1), s = sOf(u), nrmd = (1 - s) / (1 - sFar); return { s, nrmd, x: xA + R0 * 0.45 + nrmd * (xB - xA - R0 * 0.75), y: cy0 + 26 - nrmd * 56, r: R0 * s }; };
  const AX = [...Array(T)].map((_, t) => ax(t));
  const axF = tf => { const t0 = Math.max(0, Math.min(T - 2, Math.floor(tf))), f = tf - t0, a = AX[t0], b = AX[t0 + 1]; return { s: a.s + (b.s - a.s) * f, nrmd: a.nrmd + (b.nrmd - a.nrmd) * f, x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, r: a.r + (b.r - a.r) * f }; };
  const tubeAt = (tf, a, phi) => { const g = typeof tf === 'number' && Number.isInteger(tf) ? AX[tf] : axF(tf), al = a + phi; return [g.x + g.r * 0.3 * Math.cos(al), g.y - g.r * Math.sin(al), Math.cos(al), g.nrmd]; };
  // strip geometry
  const sxL = 92, sxR = w - 40, scy = top + (h - top) / 2 + 4, Hs = Math.min(175, (h - top) * 0.42);
  const stripAt = (tf, a) => [sxL + tf * (sxR - sxL) / (T - 1), scy - (wrap(a) / Math.PI) * Hs, 1, 0];
  const key = `${Math.round(Rs * dpr)}|${DARK}|${MODE}`; if (sphCacheKey !== key) { sphCache = [...Array(T)].map((_, t) => sphereImage(t, 0, Rs * dpr)); sphCacheKey = key; }
  ctx.fillStyle = C.ink; ctx.font = '600 15px system-ui'; ctx.textAlign = 'left'; ctx.fillText(`${st + 1} · ${TITLES[st]}`, 10, 18);
  const Q0 = M.Q[0], N0 = M.NAT[0];
  const phiAt = tt => 0.35 * (Math.min(Math.max(tt, START[2]), START[3]) - START[2]); // rotate while the tube is shown, then hold for the unroll
  // ---------- stages 1-2: spheres and the circles flying into the tunnel ----------
  if (st <= 1) {
    const fade = st === 0 ? 1 : 1 - 0.9 * ease(clamp(p / 0.4)), e2 = st === 1 ? ease(clamp((p - 0.35) / 0.65)) : 0;
    for (let t = 0; t < T; t++) { const [x, y] = sc(t); ctx.globalAlpha = fade; ctx.drawImage(sphCache[t], x - Rs - 1 / dpr, y - Rs - 1 / dpr, sphCache[t].width / dpr, sphCache[t].height / dpr); ctx.globalAlpha = 1;
      if (e2 < 0.05) { ctx.fillStyle = C.muted; ctx.font = '12px system-ui'; ctx.textAlign = 'center'; ctx.fillText('t = ' + (t + 1), x, y + Rs + 12); } }
    for (let t = T - 1; t >= 0; t--) {
      const pts = M.th.map(a => { const v = M.toView(M.circ(a)), [c0, c1] = sc(t), s0 = [c0 + Rs * v[0], c1 - Rs * v[1], v[2]]; if (st === 0) return s0; const q = tubeAt(t, a, 0); return [s0[0] + (q[0] - s0[0]) * e2, s0[1] + (q[1] - s0[1]) * e2, s0[2] + (q[2] - s0[2]) * e2]; });
      const lw = st === 0 ? 4 : 2.6 + 2.5 * e2 * AX[t].s; ctx.lineCap = 'round';
      for (let pass = 0; pass < 2; pass++) for (let i = 0; i < NT; i++) { const a = pts[i], b = pts[(i + 1) % NT], front = (a[2] + b[2]) / 2 > 0 || e2 > 0.6;
        if (pass === 0) { if (!front) continue; ctx.strokeStyle = hexA(C.ink, 0.75); ctx.lineWidth = lw + 1.2; ctx.setLineDash([]); }
        else { ctx.strokeStyle = front ? valCol(Q0[t][i], N0[t]) : hexA(C.ink, 0.3); ctx.lineWidth = front ? lw : 1; ctx.setLineDash(front ? [] : [2, 3]); }
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
      ctx.lineCap = 'butt'; }
    ctx.setLineDash([]);
    setRO(st, null); return;
  }
  // ---------- stages 3-6: surface quads ----------
  let eU = 0, Qa = Q0, Qb = Q0, Na = N0, Nb = N0, mix = 0, path = null, history = -1, swapE = 0, kNow = 0, marker = -1, sigma = 0;
  const phi = phiAt(tau);
  if (st === 3) { eU = ease(p); setRO(st, null); }
  if (st === 4) { // RL on the flattened tube: K iterations, each = rollout sweep, then improvement
    eU = 1; const per = DUR[4] / M.K, kf = (tau - START[4]) / per, k = Math.min(M.K - 1, Math.floor(kf)), fr = kf - k;
    const sweep = clamp(fr / 0.62), imp = ease(clamp((fr - 0.62) / 0.38));
    kNow = k + imp; mix = imp; Qa = M.Q[k]; Qb = M.Q[k + 1]; Na = M.NAT[k]; Nb = M.NAT[k + 1];
    path = pathAt(k, imp); marker = imp > 0 ? -1 : sweep * (T - 1); history = k;
    if (HEAT === 'est') { // what the learner sees: the critic's estimate, updated behind the rollout marker
      const cur = M.EST[k], prev = k ? M.EST[k - 1] : null, nat = M.NAT[k];
      const Qd = nat.map((nv, t) => { const wgt = imp > 0 ? 1 : clamp(marker - t + 0.5); return cur[t].map((cv, i) => nv + (prev ? prev[t][i] : 0) * (1 - wgt) + cv * wgt); });
      Na = Nb = nat; mix = 0; Qa = Qb = Qd;
      if (k === 0 && fr < 0.12) { Qa = M.Q[0]; Na = M.NAT[0]; Qb = Qd; mix = ease(fr / 0.12); } }
    sigma = 0.12 + 0.95 * (1 - kNow / M.K);
    if (STATIC) { const nat = M.NAT[M.K - 1]; Qa = Qb = HEAT === 'est' ? nat.map((nv, t) => M.EST[M.K - 1][t].map(v => nv + v)) : M.Q[M.K]; Na = Nb = HEAT === 'est' ? nat : M.NAT[M.K]; mix = 0; path = M.paths[M.K]; marker = -1; sigma = 0.12; kNow = M.K; history = M.K; }
    setRO(st, M.succ[k] + (M.succ[k + 1] - M.succ[k]) * imp, kNow); }
  if (st === 2) setRO(st, null);
  const appear = st === 2 ? ease(clamp(p / 0.4)) : 1;
  const PT = (tf, a) => { const q = tubeAt(tf, a, phi); if (eU === 0) return q; const s = stripAt(tf, a); return [q[0] + (s[0] - q[0]) * eU, q[1] + (s[1] - q[1]) * eU, q[2] + (1 - q[2]) * eU, q[3] * (1 - eU)]; };
  const quads = [], step = 2;
  for (let t = 0; t < T - 1; t++) for (let i = 0; i < NT; i += step) { const j = (i + step) % NT; if (j < i && eU > 0.5) continue;
    const a = PT(t, M.th[i]), b = PT(t + 1, M.th[i]), c = PT(t + 1, M.th[j]), d = PT(t, M.th[j]);
    const facing = (a[2] + b[2] + c[2] + d[2]) / 4, far = (a[3] + b[3]) / 2;
    const va = (Qa[t][i] + Qa[t + 1][i] + Qa[t][j] + Qa[t + 1][j]) / 4, vb = (Qb[t][i] + Qb[t + 1][i] + Qb[t][j] + Qb[t + 1][j]) / 4, na = (Na[t] + Na[t + 1]) / 2, nb = (Nb[t] + Nb[t + 1]) / 2;
    quads.push({ a, b, c, d, facing, far, t, v: va * (1 - mix) + vb * mix, n: na * (1 - mix) + nb * mix }); }
  quads.sort((q1, q2) => (q1.facing >= 0) - (q2.facing >= 0) || q2.t - q1.t);
  const pg = hex2rgb(C.page);
  quads.forEach(q => { const col = valRGB(q.v, q.n), lit = eU > 0 ? 1 : 0.62 + 0.38 * Math.max(0, q.facing), fog = 0.35 * q.far * (1 - eU);
    const rgb = col.map((x, i) => (x * lit) * (1 - fog) + pg[i] * fog), alpha = (q.facing < 0 && eU < 1 ? 0.3 + 0.7 * eU : 1) * appear;
    ctx.fillStyle = `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},${alpha})`;
    ctx.beginPath(); ctx.moveTo(q.a[0], q.a[1]); ctx.lineTo(q.b[0], q.b[1]); ctx.lineTo(q.c[0], q.c[1]); ctx.lineTo(q.d[0], q.d[1]); ctx.closePath(); ctx.fill(); ctx.strokeStyle = ctx.fillStyle; ctx.lineWidth = 0.6; ctx.stroke(); });
  // tunnel rim at decision 1 and axis labels
  if (eU < 0.5) { ctx.globalAlpha = (1 - eU * 2) * appear; ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.beginPath(); for (let i = 0; i <= 64; i++) { const q = tubeAt(0, -Math.PI + 2 * Math.PI * i / 64, phi); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); } ctx.stroke();
    ctx.fillStyle = C.ink2; ctx.font = '13px system-ui'; ctx.textAlign = 'center'; ctx.fillText('decision 1', AX[0].x, AX[0].y + AX[0].r + 18); ctx.fillText('decision 32', AX[T - 1].x, AX[T - 1].y + AX[T - 1].r + 16);
    ctx.fillText('decision t →  (into the distance)', (AX[0].x + AX[T - 1].x) / 2 + 40, h - 6); ctx.globalAlpha = 1; }
  if (eU > 0.5) { ctx.globalAlpha = clamp((eU - 0.5) * 2); ctx.textAlign = 'right'; ctx.fillStyle = C.muted; ctx.font = '13px system-ui';
    [[Math.PI, 'π'], [Math.PI / 2, 'π/2'], [0, '0'], [-Math.PI / 2, '−π/2'], [-Math.PI + 1e-6, '−π']].forEach(([a, l]) => ctx.fillText(l, sxL - 8, scy - (a / Math.PI) * Hs + 4));
    ctx.save(); ctx.translate(16, scy); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillStyle = C.ink2; ctx.fillText('position in the subspace  û  (wraps around)', 0, 0); ctx.restore();
    ctx.textAlign = 'center'; [1, 8, 16, 24, 32].forEach(t => { ctx.fillStyle = C.muted; ctx.fillText(t, stripAt(t - 1, 0)[0], scy + Hs + 14); }); ctx.fillStyle = C.ink2; ctx.fillText('decision t →', (sxL + sxR) / 2, h - 6); ctx.globalAlpha = 1; }
  // ---------- exploration cloud (stage 4) ----------
  if (st === 4 && sigma > 0) {
    for (let t = 0; t < T; t++) for (let j = 0; j < M.EX[t].length; j++) {
      const a = path[t] + sigma * M.EX[t][j], q = PT(t, a); if (q[2] < 0) continue;
      const near = marker >= 0 ? Math.exp(-((t - marker) ** 2) / 3) : 0, rr = 1.8 + 2.4 * near;
      ctx.beginPath(); ctx.arc(q[0], q[1], rr, 0, 7); ctx.fillStyle = hexA(C.ink, 0.35 + 0.5 * near); ctx.fill(); } }
  // ---------- paths ----------
  const drawPath3D = (pth, col, width, faintBack, dots) => { // on the surface, sampling between decisions
    let prev = null; for (let t = 0; t < T - 1; t++) for (let sbt = 0; sbt < 4; sbt++) {
      const f0 = sbt / 4, f1 = (sbt + 1) / 4, a0 = angLerp(pth[t], pth[t + 1], f0), a1 = angLerp(pth[t], pth[t + 1], f1);
      const q0 = PT(t + f0, a0), q1 = PT(t + f1, a1), front = (q0[2] + q1[2]) / 2 >= 0 || eU > 0.5;
      if (eU > 0.5 && Math.abs(wrap(a1) - wrap(a0)) > Math.PI) continue;
      const s = axF(t + f0).s, lw = eU > 0.5 ? width : width * (0.5 + 0.5 * s) + 0.5;
      if (front) { ctx.strokeStyle = C.surf; ctx.lineWidth = lw + 3; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q1[0], q1[1]); ctx.stroke(); }
      ctx.strokeStyle = front ? col : hexA(C.ink, faintBack); ctx.lineWidth = front ? lw : 1; ctx.setLineDash(front ? [] : [3, 3]); ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q1[0], q1[1]); ctx.stroke(); }
    ctx.setLineDash([]);
    if (dots) pth.forEach((a, t) => { const q = PT(t, a); if (q[2] < 0 && eU < 0.5) return; const r = eU > 0.5 ? 3.4 : 2 + 2.6 * AX[t].s; ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.fillStyle = C.r; ctx.fill(); ctx.strokeStyle = C.surf; ctx.lineWidth = 1.2; ctx.stroke(); }); };
  if (history >= 0) for (let k = 0; k < Math.min(history, M.K); k++) drawPath3D(M.paths[k], hexA(C.ink, k === 0 ? 0.5 : 0.16 + 0.05 * k), 1.3, 0.15, false);
  if (path && st >= 4) drawPath3D(path, C.ink, 2.6, 0.3, true);
  // rollout marker with a short trail
  if (st === 4 && marker >= 0) {
    for (let k = 0; k < 10; k++) { const tf = marker - k * 0.25; if (tf < 0) break; const t0 = Math.min(T - 2, Math.floor(tf)), f = tf - t0, a = angLerp(path[t0], path[t0 + 1], f), q = PT(tf, a);
      if (q[2] < 0) continue; ctx.beginPath(); ctx.arc(q[0], q[1], (7 - k * 0.55), 0, 7); ctx.fillStyle = hexA(C.r, 0.85 - k * 0.08); ctx.fill(); }
    ctx.fillStyle = C.ink2; ctx.font = '13px system-ui'; ctx.textAlign = 'right'; ctx.fillText('green marker: one rollout of the current policy', sxR, 36); }
  if (st === 4) { ctx.fillStyle = C.ink2; ctx.font = '13px system-ui'; ctx.textAlign = 'left'; ctx.fillText('grey dots: exploration samples around the path (spread shrinks as training proceeds)', 10, 36);
    ctx.textAlign = 'right'; ctx.fillText('faint lines: earlier iterations (darkest = untrained)', sxR, scy + Hs + 30);
    ctx.textAlign = 'left'; ctx.font = '600 13px system-ui'; ctx.fillStyle = C.ink; ctx.fillText(HEAT === 'est' ? "Heat: the critic's estimate from rollouts so far (grey = not yet explored)" : 'Heat: the true advantage (the learner never sees this map)', 10, 52); }
  if (st === 3 && p > 0.6) { ctx.globalAlpha = clamp((p - 0.6) / 0.4); ctx.fillStyle = C.ink2; ctx.font = '13px system-ui'; ctx.textAlign = 'right'; ctx.fillText('top and bottom edges are the same point: the strip wraps around', sxR, scy + Hs + 30); ctx.globalAlpha = 1; }
}
function setRO(st, s, k) {
  document.getElementById('roPath').textContent = s === null ? '—' : (s * 100).toFixed(0) + '%';
  document.getElementById('roStageLab').textContent = st === 4 ? 'Policy iteration' : 'Stage';
  document.getElementById('roStage').textContent = st === 4 ? `k = ${k.toFixed(1)} of ${M.K}` : STAGE_NAMES[st];
}
document.querySelectorAll('#heatSeg button').forEach(b => b.onclick = () => { HEAT = b.dataset.heat; document.querySelectorAll('#heatSeg button').forEach(x => x.setAttribute('aria-pressed', x === b)); });
document.querySelectorAll('.modeSeg button').forEach(b => b.onclick = () => { MODE = b.dataset.cm; document.querySelectorAll('.modeSeg button').forEach(x => x.setAttribute('aria-pressed', x.dataset.cm === MODE)); updateLegend(); invalidate(); });
function updateLegend() { const el = document.getElementById('rampSw'), lab = document.getElementById('rampLab'); if (!el) return;
  if (MODE === 'adv') { const a = divRGB(-1), m = divRGB(0), b = divRGB(1); el.style.background = `linear-gradient(90deg, rgb(${a.map(x => x | 0)}), rgb(${m.map(x => x | 0)}), rgb(${b.map(x => x | 0)}))`; lab.textContent = `advantage: value minus the base-policy average at that decision (amber lower, violet higher; ±${(M.AMAX * 100).toFixed(0)} points at full colour)`; }
  else { el.style.background = `linear-gradient(90deg, ${RAMP.join(',')})`; lab.textContent = `absolute value Q, ${(M.ABSLO * 100).toFixed(0)}% → ${(M.ABSHI * 100).toFixed(0)}% success, one scale for all stages. Within one decision values differ by only a few points, so they look faint here; the strip brightens as training improves.`; } }
updateLegend();
/* ---------- Figure 2: one cross-section ---------- */
const tSel = document.getElementById('tSel'), kSel = document.getElementById('kSel');
[tSel, kSel].filter(Boolean).forEach(el => el.addEventListener('input', () => { drawSph(); drawRing(); }));
function drawSph() { if (!document.getElementById('sph')) return;
  const t = +tSel.value, k = +kSel.value; document.getElementById('tSelV').textContent = t + 1; document.getElementById('kSelV').textContent = k;
  const cv = document.getElementById('sph'), { ctx, w, h, dpr } = fit(cv); const Rs = Math.min(w, h) * 0.42, cx = w / 2, cyy = h / 2;
  const img = sphereImage(t, k, Rs * dpr); ctx.drawImage(img, cx - Rs - 1 / dpr, cyy - Rs - 1 / dpr, img.width / dpr, img.height / dpr);
  const pts = M.th.map(a => M.toView(M.circ(a)));
  for (let i = 0; i < M.NT; i++) { const a = pts[i], b = pts[(i + 1) % M.NT], front = a[2] + b[2] > 0; ctx.strokeStyle = front ? C.ink : hexA(C.ink, 0.45); ctx.lineWidth = front ? 2 : 1.2; ctx.setLineDash(front ? [] : [3, 3]);
    ctx.beginPath(); ctx.moveTo(cx + Rs * a[0], cyy - Rs * a[1]); ctx.lineTo(cx + Rs * b[0], cyy - Rs * b[1]); ctx.stroke(); }
  ctx.setLineDash([]);
  const v = M.toView(M.circ(M.paths[k][t])); ctx.beginPath(); ctx.arc(cx + Rs * v[0], cyy - Rs * v[1], 6, 0, 7); ctx.fillStyle = v[2] > 0 ? C.r : C.surf; ctx.fill(); ctx.strokeStyle = C.r; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = C.ink2; ctx.font = '13px system-ui'; ctx.textAlign = 'left'; ctx.fillText(`decision ${t + 1}${M.strong[t] ? ' (higher contrast)' : ''}`, 6, 14);
}
function drawRing() { if (!document.getElementById('ring')) return;
  const t = +tSel.value, k = +kSel.value, q = M.Q[k][t];
  const cv = document.getElementById('ring'), { ctx, w, h } = fit(cv); const pad = { l: 40, r: 10, t: 22, b: 30 };
  const natv = M.NAT[k][t], qmin = Math.min(natv, ...q), qmax = Math.max(natv, ...q), span = Math.max(0.04, qmax - qmin), lo = qmin - 0.15 * span, hi = qmax + 0.15 * span;
  const X = a => pad.l + (a + Math.PI) / (2 * Math.PI) * (w - pad.l - pad.r), Y = v => h - pad.b - 12 - (v - lo) / (hi - lo) * (h - pad.t - pad.b - 12);
  ctx.font = '13px system-ui'; ctx.lineWidth = 1;
  const stp = span > 0.12 ? 0.05 : span > 0.05 ? 0.02 : 0.01; for (let v = Math.ceil(lo / stp) * stp; v <= hi; v += stp) { ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke(); ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.fillText((v * 100).toFixed(0) + '%', pad.l - 6, Y(v) + 4); }
  ctx.textAlign = 'center'; [[-Math.PI, '−π'], [-Math.PI / 2, '−π/2'], [0, '0'], [Math.PI / 2, 'π/2'], [Math.PI, 'π']].forEach(([a, l]) => ctx.fillText(l, X(a), h - pad.b + 14));
  ctx.fillStyle = C.ink2; ctx.fillText('position in the subspace û', (pad.l + w - pad.r) / 2, h - 4);
  for (let i = 0; i < M.NT; i++) { ctx.fillStyle = valCol(q[i], M.NAT[k][t]); ctx.fillRect(X(M.th[i]), h - pad.b - 8, (w - pad.l - pad.r) / M.NT + 1, 8); }
  const nat = M.NAT[k][t], sm = q.reduce((a, b) => a + b, 0) / M.NT;
  ctx.setLineDash([6, 4]); ctx.strokeStyle = C.muted; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(pad.l, Y(nat)); ctx.lineTo(w - pad.r, Y(nat)); ctx.stroke();
  ctx.setLineDash([2, 3]); ctx.strokeStyle = C.ink2; ctx.beginPath(); ctx.moveTo(pad.l, Y(sm)); ctx.lineTo(w - pad.r, Y(sm)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2; ctx.beginPath(); q.forEach((v, i) => i ? ctx.lineTo(X(M.th[i]), Y(v)) : ctx.moveTo(X(M.th[i]), Y(v))); ctx.stroke();
  const a = M.paths[k][t], va = q[Math.round(((wrap(a) + Math.PI) / (2 * Math.PI)) * M.NT) % M.NT];
  ctx.beginPath(); ctx.arc(X(wrap(a)), Y(va), 5.5, 0, 7); ctx.fillStyle = C.r; ctx.fill(); ctx.strokeStyle = C.surf; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = C.ink2; ctx.textAlign = 'left'; ctx.font = '13px system-ui'; ctx.fillText(`native ${(nat * 100).toFixed(1)}% · subspace average ${(sm * 100).toFixed(1)}% · actor ${(va * 100).toFixed(1)}%  (axis zoomed to this decision)`, pad.l + 4, pad.t - 6);
}

/* ---------- Figure 3: success per iteration ---------- */
function drawIter() { if (!document.getElementById('iter')) return;
  const cv = document.getElementById('iter'), { ctx, w, h } = fit(cv); const pad = { l: 44, r: 20, t: 12, b: 32 }, n = M.K + 2;
  const X = i => pad.l + i / (n - 1) * (w - pad.l - pad.r), Y = v => h - pad.b - v * (h - pad.t - pad.b);
  ctx.font = '13px system-ui'; ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(v => { ctx.strokeStyle = v ? C.grid : C.axis; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke(); ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.fillText(v * 100 + '%', pad.l - 6, Y(v) + 4); });
  ctx.textAlign = 'center'; for (let i = 0; i <= M.K; i++) ctx.fillText(i, X(i), h - pad.b + 14); ctx.fillText('swap to P′', X(M.K + 1), h - pad.b + 14);
  ctx.fillStyle = C.ink2; ctx.fillText('policy-iteration step k', X(M.K / 2), h - 4);
  const hl = (v, dash, col) => { ctx.setLineDash(dash); ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke(); ctx.setLineDash([]); };
  hl(M.native, [6, 4], C.muted); hl(M.uni, [2, 3], C.ink2);
  ctx.fillStyle = C.ink2; ctx.textAlign = 'left'; ctx.fillText(`native ${(M.native * 100).toFixed(0)}%`, w - pad.r - 160, Y(M.native) + 14); ctx.fillText(`random direction in P ${(M.uni * 100).toFixed(0)}%`, w - pad.r - 160, Y(M.uni) - 6);
  ctx.strokeStyle = C.r; ctx.lineWidth = 2.2; ctx.beginPath(); M.succ.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))); ctx.stroke();
  const dt = (x, y, hollow) => { ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); if (hollow) { ctx.fillStyle = C.surf; ctx.fill(); ctx.strokeStyle = C.r; ctx.lineWidth = 2; ctx.stroke(); } else { ctx.fillStyle = C.r; ctx.fill(); ctx.strokeStyle = C.surf; ctx.lineWidth = 1.5; ctx.stroke(); } };
  M.succ.forEach((v, i) => dt(X(i), Y(v)));
  ctx.setLineDash([4, 4]); ctx.strokeStyle = hexA(C.r, 0.7); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(M.K), Y(M.succ[M.K])); ctx.lineTo(X(M.K + 1), Y(M.succSwap)); ctx.stroke(); ctx.setLineDash([]);
  dt(X(M.K + 1), Y(M.succSwap), true);
  ctx.fillStyle = C.ink; ctx.font = '600 13.5px system-ui'; ctx.textAlign = 'center'; ctx.fillText((M.succ[M.K] * 100).toFixed(0) + '%', X(M.K), Y(M.succ[M.K]) - 10); ctx.fillText((M.succSwap * 100).toFixed(0) + '%', X(M.K + 1), Y(M.succSwap) - 10); ctx.fillText((M.succ[0] * 100).toFixed(0) + '%', X(0), Y(M.succ[0]) - 10);
}


/* ---------- Figure 2: full sphere vs subspace (exact in y = Jz) ---------- */
const HD = (() => {
  const rng = mulberry(3), g = () => { let u = 0; while (!u) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); };
  const m = 8, NC = 12, Cc = [...Array(NC)].map(() => [...Array(m)].map(g)), Sg = [...Array(NC)].map(() => rng() < 0.5 ? -1 : 1);
  const Qt = y => { let s = 0; for (let c = 0; c < NC; c++) { let d = 0; for (let k = 0; k < m; k++) d += (y[k] - Cc[c][k]) ** 2; s += Sg[c] * Math.exp(-d / 4); } return 1 / (1 + Math.exp(-2 * s)); };
  const nat = [...Array(3000)].map(() => Qt([...Array(m)].map(g))), mu = nat.reduce((a, b) => a + b) / nat.length, sd = Math.sqrt(nat.reduce((a, b) => a + (b - mu) ** 2, 0) / nat.length);
  const circles = [...Array(4)].map(() => ({ a: [...Array(m)].map(g), b: [...Array(m)].map(g) }));
  const NP = 360, th = [...Array(NP)].map((_, i) => -Math.PI + 2 * Math.PI * i / NP);
  const subspace = circles.map(c => th.map(t => Qt(c.a.map((x, k) => Math.cos(t) * x + Math.sin(t) * c.b[k]))));
  const sdOf = v => { const mm = v.reduce((a, b) => a + b) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - mm) ** 2, 0) / v.length); };
  const corrLen = v => { const n = v.length, mm = v.reduce((a, b) => a + b) / n, va = v.reduce((a, b) => a + (b - mm) ** 2, 0) / n; for (let L = 1; L < n / 2; L++) { let c = 0; for (let i = 0; i < n; i++) c += (v[i] - mm) * (v[(i + L) % n] - mm); if (c / n / va < 0.5) return 2 * Math.PI * L / n; } return Math.PI; };
  const sens = [0, ...[...Array(m - 1)].map(g)]; // a random point orthogonal to the sensitive direction e0
  return { m, Qt, mu, sd, th, subspace, subspaceSd: subspace.map(sdOf), corr: subspace.map(corrLen), sens, N: 50000, t: 0 };
})();
document.querySelectorAll('#hdSeg button').forEach(b => b.onclick = () => { HD.N = +b.dataset.n; document.querySelectorAll('#hdSeg button').forEach(x => x.setAttribute('aria-pressed', x === b)); });
function drawHD(dt) { if (!document.getElementById('hd-a')) return;
  if (!STATIC) HD.t += dt;
  const f = STATIC ? -1 : (HD.t * 0.12) % 1;
  document.getElementById('hdNat').textContent = HD.sd.toFixed(2);
  document.getElementById('hdSl').textContent = `${Math.min(...HD.subspaceSd).toFixed(2)}–${Math.max(...HD.subspaceSd).toFixed(2)}`;
  document.getElementById('hdCorr').textContent = `${Math.min(...HD.corr).toFixed(2)}–${Math.max(...HD.corr).toFixed(2)} rad`;
  document.getElementById('hdStep').textContent = (1 / Math.sqrt(HD.N)).toFixed(4) + ' rad';
  const panel = (id, xs, curves, xmin, xmax, fmtX, note) => {
    const cv = document.getElementById(id), { ctx, w, h } = fit(cv), pad = { l: 40, r: 10, t: 14, b: 32 };
    const X = x => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r), Y = v => h - pad.b - v * (h - pad.t - pad.b);
    ctx.fillStyle = css('--band'); ctx.fillRect(pad.l, Y(HD.mu + HD.sd), w - pad.l - pad.r, Y(HD.mu - HD.sd) - Y(HD.mu + HD.sd));
    ctx.font = '13px system-ui'; [0, 0.25, 0.5, 0.75, 1].forEach(v => { ctx.strokeStyle = v ? C.grid : C.axis; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke(); ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.fillText(v, pad.l - 6, Y(v) + 4); });
    [xmin, (xmin + xmax) / 2, xmax].forEach((x, i) => { ctx.fillStyle = C.muted; ctx.textAlign = ['left', 'center', 'right'][i]; ctx.fillText(fmtX(x), X(x), h - pad.b + 14); }); ctx.textAlign = 'center';
    ctx.fillStyle = C.ink2; ctx.fillText(note, (pad.l + w - pad.r) / 2, h - 4);
    curves.forEach((v, ci) => { const bold = ci === 0; ctx.strokeStyle = bold ? C.ink : hexA(C.muted, 0.7); ctx.lineWidth = bold ? 2.4 : 1.2; ctx.beginPath(); v.forEach((y, i) => i ? ctx.lineTo(X(xs[i]), Y(y)) : ctx.moveTo(X(xs[i]), Y(y))); ctx.stroke(); });
    if (f >= 0) { const i = Math.min(xs.length - 1, Math.floor(f * xs.length)), v = curves[0][i]; ctx.beginPath(); ctx.arc(X(xs[i]), Y(v), 5.5, 0, 7); ctx.fillStyle = C.r; ctx.fill(); ctx.strokeStyle = C.surf; ctx.lineWidth = 1.5; ctx.stroke(); }
    return { X, Y, pad, w, h, ctx };
  };
  panel('hd-a', HD.th, HD.subspace, -Math.PI, Math.PI, x => Math.abs(x) < 1e-9 ? '0' : (x < 0 ? '−π' : 'π'), 'angle along the circle (rad): the full circle');
  const sN = Math.sqrt(HD.N), half = 12 / sN, NPb = 400, xs = [...Array(NPb)].map((_, i) => -half + 2 * half * i / (NPb - 1));
  const vb = xs.map(t => HD.Qt(HD.sens.map((x, k) => Math.cos(t) * x + (k === 0 ? Math.sin(t) * sN : 0))));
  const P = panel('hd-b', xs, [vb], -half, half, x => (Math.abs(x) < 1e-12 ? '0' : x.toFixed(4)), `angle (rad): this panel spans only ${(2 * half).toFixed(4)} rad`);
  const ctx = P.ctx; ctx.strokeStyle = C.axis; ctx.setLineDash([3, 3]); ctx.lineWidth = 1; [-1, 1].forEach(s => { const x = P.X(s / sN); ctx.beginPath(); ctx.moveTo(x, P.pad.t); ctx.lineTo(x, P.h - P.pad.b); ctx.stroke(); }); ctx.setLineDash([]);
  ctx.fillStyle = C.ink2; ctx.font = '13px system-ui'; ctx.textAlign = 'left'; ctx.fillText('±1/√N', P.X(1 / sN) + 4, P.pad.t + 10);
}
/* ---------- loop ---------- */
let last = performance.now();
function frame(now) { const dt = Math.min(0.05, (now - last) / 1000); last = now; const box = document.getElementById('tube').getBoundingClientRect(); if (box.bottom > 0 && box.top < innerHeight && !document.hidden) drawTube(dt); requestAnimationFrame(frame); }
let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(invalidate, 150); });
invalidate(); requestAnimationFrame(frame);
})();
