// The play page: sends keys and mouse to wm/play.py, shows the frames it sends back.
(() => {
"use strict";
const $ = id => document.getElementById(id);
const token = new URLSearchParams(location.search).get("token") || "";
const S = {
  levels: [], level: -1, fps: 10, hasData: false, open: false, started: false, locked: false, paused: false,
  vhs: true, hud: true, muted: false, rec: false, light: 0, sens: 0.0025, invX: false, invY: false, blend: true,
  opts: {}, lastI: -1, info: {}, ms: null, n: 0, lat: 0, times: [], aspect: 16 / 9,
};
const inp = { dx: 0, dy: 0, keys: new Set() };
let ws = null;
window.__noclip = S;

// ---------------------------------------------------------------- picture
const canvas = $("gl");
const gl = canvas.getContext("webgl", { antialias: false, alpha: false });
const VS = "attribute vec2 p; varying vec2 uv; void main(){ uv = p*0.5+0.5; uv.y = 1.0-uv.y; gl_Position = vec4(p,0.,1.); }";
const FS = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 uv; uniform sampler2D a, b; uniform float mixv, time, vhs, lines;
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
vec3 frame(vec2 q){ return mix(texture2D(a, q).rgb, texture2D(b, q).rgb, mixv); }
void main(){
  vec2 c = uv - 0.5; float r2 = dot(c, c);
  vec2 q = 0.5 + c*(1.0 + 0.05*vhs*r2);                                    // slight barrel
  float row = floor(q.y*lines);
  float tear = step(0.992, hash(vec2(row*0.37, floor(time*9.0))));          // a line that slips sideways now and then
  q.x += vhs*(0.0035*tear + 0.0006*sin(q.y*40.0 + time*3.0));
  float ca = vhs*0.0016*(1.0 + 3.0*r2);                                    // colours come apart towards the edges
  vec3 col = vec3(frame(q + vec2(ca, 0.)).r, frame(q).g, frame(q - vec2(ca, 0.)).b);
  col *= 1.0 - 0.10*vhs*(0.5 + 0.5*cos(q.y*lines*6.2832));                 // scan lines
  col += vhs*0.055*(hash(q*vec2(640., 360.) + fract(time)*91.7) - 0.5);    // grain
  float band = 1.0 - smoothstep(0.0, 0.03, abs(fract(q.y - time*0.07) - 0.5));
  col += vhs*0.025*band;                                                   // a slow bright band
  col *= 1.0 - vhs*0.55*smoothstep(0.18, 0.62, r2*1.6);                    // vignette
  col = mix(col, vec3(dot(col, vec3(0.3, 0.59, 0.11))), 0.08*vhs);
  if (q.x < 0. || q.x > 1. || q.y < 0. || q.y > 1.) col = vec3(0.);
  gl_FragColor = vec4(col, 1.);
}`;
function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
const U = {}; for (const n of ["a", "b", "mixv", "time", "vhs", "lines"]) U[n] = gl.getUniformLocation(prog, n);
const tex = [0, 1].map(() => {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0]));
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  return t;
});
let cur = 0, tFrame = 0, lines = 288;

function resize() {
  const w = innerWidth, h = innerHeight, s = Math.min(w / S.aspect, h);
  const cw = Math.round(s * S.aspect), ch = Math.round(s), dpr = Math.min(devicePixelRatio || 1, 2);
  const st = $("stage"); st.style.width = cw + "px"; st.style.height = ch + "px"; st.style.fontSize = (cw / 100) + "px";
  canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
  gl.viewport(0, 0, canvas.width, canvas.height);
}
addEventListener("resize", resize); resize();

function draw(now) {
  const blendMs = 0.65 * 1000 / S.fps;
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex[1 - cur]); gl.uniform1i(U.a, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, tex[cur]); gl.uniform1i(U.b, 1);
  gl.uniform1f(U.mixv, S.blend ? Math.min(1, (now - tFrame) / blendMs) : 1);
  gl.uniform1f(U.time, (now / 1000) % 1000); gl.uniform1f(U.vhs, S.vhs ? 1 : 0); gl.uniform1f(U.lines, lines);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// ---------------------------------------------------------------- frames from the server
const utf8 = new TextDecoder();
async function onFrame(buf) {
  const n = new DataView(buf).getUint32(0, true);
  const hdr = JSON.parse(utf8.decode(new Uint8Array(buf, 4, n)));
  const bmp = await createImageBitmap(new Blob([new Uint8Array(buf, 4 + n)], { type: "image/jpeg" }));
  if (hdr.i <= S.lastI) { bmp.close(); return; }   // decoded out of order
  S.lastI = hdr.i;
  const now = performance.now();
  cur = 1 - cur; gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex[cur]);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, bmp);
  if (Math.abs(bmp.width / bmp.height - S.aspect) > 0.01) { S.aspect = bmp.width / bmp.height; resize(); }
  lines = bmp.height; bmp.close(); tFrame = now;
  S.times.push(now); while (S.times.length > 1 && now - S.times[0] > 2000) S.times.shift();
  if (hdr.ct > 0) { const l = now - hdr.ct; S.lat = S.lat ? 0.85 * S.lat + 0.15 * l : l; }
  S.ms = hdr.ms; S.n = hdr.n; S.mem = hdr.mem || 0; S.near = !!hdr.near;
  if (hdr.ev === "hit") audio.scream();
  if (hdr.ev === "dead") { audio.death(); toast("you died", 3000); }
  if (!!hdr.rec !== S.rec) setRec(!!hdr.rec);
  if (hdr.lvl !== S.level) setLevel(hdr.lvl);
}
function setLevel(id) {
  const first = S.level < 0; S.level = id;
  const name = S.levels[id] || ("Level " + id);
  const t = $("title"); t.innerHTML = ""; t.append(name);
  const sm = document.createElement("small"); sm.textContent = first ? "you are here" : "you have noclipped"; t.append(sm);
  t.classList.add("show"); clearTimeout(setLevel.h); setLevel.h = setTimeout(() => t.classList.remove("show"), 2600);
  audio.level(id, !first);
  for (const b of $("levels").children) b.classList.toggle("here", +b.dataset.id === id);
}
function hud() {
  const t = S.times, fps = t.length > 1 ? (t.length - 1) * 1000 / (t[t.length - 1] - t[0]) : 0, m = S.ms;
  const l = [`${S.levels[S.level] || "…"}`,
             `${fps.toFixed(1)} frames/s · ${S.lat.toFixed(0)} ms from input to picture`];
  if (m) l.push(`server ${m.tot.toFixed(0)} ms: context ${m.ctx.toFixed(0)} · denoise ${m.den.toFixed(0)} · decode ${m.dec.toFixed(0)} · jpeg ${m.jpg.toFixed(0)}`);
  l.push(`sees the last ${S.n} frames${S.opts.mem ? ` and ${S.mem} from memory` : ""} · ${S.opts.steps ?? "?"} steps · guidance ${S.opts.guide ?? "?"}${S.opts.exact ? " · exact" : ""}${S.near ? " · something is near" : ""}`);
  if (S.info.ckpt) l.push(`${S.info.ckpt} · ${S.info.params_m}M parameters`);
  $("hud").textContent = l.join("\n");
  $("hud").classList.toggle("hidden", !S.hud || !S.started);
  $("hint").textContent = !S.open ? "not connected" : S.paused ? "paused · P to go on" : S.locked ? ""
    : S.nolock ? "hold the mouse button and drag to look around (this browser does not let the page capture the mouse)"
    : "click to look around · Esc frees the mouse";
  $("hint").classList.toggle("hidden", !S.started);
}
setInterval(hud, 250);

// ---------------------------------------------------------------- the tape's clock
const t0 = performance.now(), tape0 = new Date(1996, 8, 23, 2, 13, 7).getTime();
function stamp() {
  const d = new Date(tape0 + performance.now() - t0), p = n => String(n).padStart(2, "0");
  const mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][d.getMonth()];
  const h = d.getHours() % 12 || 12;
  $("clock").textContent = `${d.getHours() < 12 ? "AM" : "PM"} ${h}:${p(d.getMinutes())}:${p(d.getSeconds())}  ${mon}. ${d.getDate()} ${d.getFullYear()}`;
  $("stamp").classList.toggle("hidden", !S.vhs || !S.started);
}
setInterval(stamp, 500);
function setRec(on) {
  S.rec = on; $("stamp").classList.toggle("rec", on);
  $("recdot").innerHTML = on ? "&#9679;" : "&#9654;"; $("mode").textContent = on ? "REC" : "PLAY";
}
function toast(msg, ms = 2600) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), ms);
}

// ---------------------------------------------------------------- sound, made here: hum, air, a low drone, steps
const audio = (() => {
  let ctx = null, master, humF, humG, o1, o2, air, airF, airG, drone, droneG, white;
  const rnd = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  function noise(seconds, brown) {
    const n = Math.floor(ctx.sampleRate * seconds), b = ctx.createBuffer(1, n, ctx.sampleRate), d = b.getChannelData(0);
    let last = 0;
    for (let i = 0; i < n; i++) { const w = Math.random() * 2 - 1; if (brown) { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; } else d[i] = w; }
    return b;
  }
  function init() {
    if (ctx) { ctx.resume(); return; }
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createGain(); master.gain.value = S.muted ? 0 : 0.6; master.connect(ctx.destination);
    humF = ctx.createBiquadFilter(); humF.type = "lowpass"; humF.frequency.value = 700;
    humG = ctx.createGain(); humG.gain.value = 0.04; humF.connect(humG); humG.connect(master);
    o1 = ctx.createOscillator(); o1.type = "sawtooth"; o1.frequency.value = 100; o1.connect(humF); o1.start();
    o2 = ctx.createOscillator(); o2.type = "square"; o2.frequency.value = 50.4;
    const g2 = ctx.createGain(); g2.gain.value = 0.35; o2.connect(g2); g2.connect(humF); o2.start();
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.13; const lg = ctx.createGain(); lg.gain.value = 0.012;
    lfo.connect(lg); lg.connect(humG.gain); lfo.start();
    air = ctx.createBufferSource(); air.buffer = noise(4, true); air.loop = true;
    airF = ctx.createBiquadFilter(); airF.type = "lowpass"; airF.frequency.value = 500;
    airG = ctx.createGain(); airG.gain.value = 0.22; air.connect(airF); airF.connect(airG); airG.connect(master); air.start();
    drone = ctx.createOscillator(); drone.type = "sine"; drone.frequency.value = 55;
    droneG = ctx.createGain(); droneG.gain.value = 0; drone.connect(droneG); droneG.connect(master); drone.start();
    white = noise(1, false);
    if (S.level >= 0) level(S.level, false);
  }
  function level(id, glitch) {   // every level has its own hum, air and drone, the same each time you come back
    if (!ctx) return;
    const r = rnd(id * 7919 + 13), t = ctx.currentTime, mains = [50, 60, 55, 48][Math.floor(r() * 4)];
    o1.frequency.setTargetAtTime(2 * mains * (1 + (r() - 0.5) * 0.01), t, 0.5);
    o2.frequency.setTargetAtTime(mains * (1 + (r() - 0.5) * 0.02), t, 0.5);
    humF.frequency.setTargetAtTime(300 + r() * 1400, t, 0.8);
    humG.gain.setTargetAtTime(0.015 + r() * 0.05, t, 0.8);
    airF.frequency.setTargetAtTime(220 + r() * 1200, t, 0.8);
    airG.gain.setTargetAtTime(0.1 + r() * 0.25, t, 0.8);
    drone.frequency.setTargetAtTime(36 + r() * 34, t, 1.0);
    droneG.gain.setTargetAtTime(r() < 0.6 ? 0.03 + r() * 0.1 : 0, t, 1.0);
    if (glitch) burst(0.7, 2600, 110, 0.5, "bandpass");
  }
  function burst(seconds, f0, f1, gain, type) {
    if (!ctx) return;
    const s = ctx.createBufferSource(); s.buffer = white; s.loop = true;
    const f = ctx.createBiquadFilter(); f.type = type; f.Q.value = 1.2; const g = ctx.createGain(), t = ctx.currentTime;
    f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + seconds);
    g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0008, t + seconds);
    s.connect(f); f.connect(g); g.connect(master); s.start(t); s.stop(t + seconds + 0.05);
  }
  return {
    init, level,
    step: () => burst(0.09, 170 + Math.random() * 40, 90, 0.5, "lowpass"),
    click: () => burst(0.03, 3000, 1500, 0.15, "bandpass"),
    scream: () => { burst(0.9, 2400, 500, 1.6, "bandpass"); burst(0.5, 900, 3200, 0.9, "bandpass"); },
    death: () => burst(2.4, 400, 40, 1.2, "lowpass"),
    mute: m => { if (ctx) master.gain.setTargetAtTime(m ? 0 : 0.6, ctx.currentTime, 0.05); },
  };
})();
let stepAt = 0;
setInterval(() => {   // footsteps while a movement key is down
  const k = inp.keys, moving = S.started && !S.paused && ["KeyW", "KeyA", "KeyS", "KeyD"].some(c => k.has(c));
  const now = performance.now(), every = k.has("ShiftLeft") || k.has("ShiftRight") ? 300 : 520;
  if (moving && now - stepAt > every) { stepAt = now; audio.step(); }
}, 40);

// ---------------------------------------------------------------- connection
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws${token ? "?token=" + encodeURIComponent(token) : ""}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { S.open = true; S.lastI = -1; S.level = -1; $("status").textContent = "connected"; };
  ws.onmessage = ev => typeof ev.data === "string" ? onText(JSON.parse(ev.data)) : onFrame(ev.data);
  ws.onclose = ev => {
    S.open = false;
    if (ev.code === 4000) { $("status").textContent = "someone else connected and is playing now; Enter takes it back"; showStart(); return; }
    $("status").textContent = "no connection to the server, trying again…";
    setTimeout(connect, 2000);
  };
}
function onText(m) {
  if (m.t === "hello") {
    S.levels = m.levels; S.fps = m.fps; S.hasData = m.has_data; S.opts = m.opts; S.info = m;
    $("newData").classList.toggle("hidden", !m.has_data);
    const box = $("levels"); box.innerHTML = "";
    m.levels.forEach((name, i) => {
      const b = document.createElement("button"); b.dataset.id = i;
      if (i < 10) { const k = document.createElement("kbd"); k.textContent = i; b.append(k); }
      b.append(name); b.onclick = () => { noclip(i); hideMenu(true); }; box.append(b);
    });
    fillPanel();
  } else if (m.t === "opts") { S.opts = m.opts; fillPanel(); }
  else if (m.t === "rec") { setRec(!!m.on); toast(m.on ? `recording to ${m.path}` : `saved ${m.path}`, 6000); }
  else if (m.t === "note") toast(m.msg, 5000);
}
setInterval(() => {   // the input state, about 30 times a second
  if (!S.open || !S.started || S.paused) return;
  const k = inp.keys, on = c => k.has(c) ? 1 : 0;
  inp.dx += (on("ArrowRight") - on("ArrowLeft")) * 0.07; inp.dy += (on("ArrowUp") - on("ArrowDown")) * 0.04;
  send({ t: "in", f: on("KeyW") - on("KeyS"), r: on("KeyD") - on("KeyA"), sp: on("ShiftLeft") | on("ShiftRight"), li: S.light,
         dx: inp.dx * (S.invX ? -1 : 1), dy: inp.dy * (S.invY ? -1 : 1), ct: performance.now() });
  inp.dx = 0; inp.dy = 0;
}, 33);
function noclip(level) { send({ t: "noclip", level }); }
function reset(mode) { send({ t: "reset", mode, level: S.level }); toast(mode === "data" ? "a new dream, from a recording" : "a new dream, from nothing"); }

// ---------------------------------------------------------------- controls
function lock() {   // capture the mouse; where the browser refuses, dragging with the button down looks around instead
  try { const p = canvas.requestPointerLock(); if (p && p.catch) p.catch(() => { S.nolock = true; }); } catch (e) { S.nolock = true; }
}
function showStart() { S.started = false; $("start").classList.remove("hidden"); if (document.pointerLockElement) document.exitPointerLock(); }
function enter() {
  if (!S.open && (!ws || ws.readyState > 1)) connect();
  $("start").classList.add("hidden"); S.started = true; audio.init(); lock();
}
$("enter").onclick = enter;
canvas.onclick = () => { if (S.started) { audio.init(); lock(); } };
canvas.addEventListener("mousedown", () => { S.drag = true; });
addEventListener("mouseup", () => { S.drag = false; });
$("stamp").onclick = () => send({ t: "rec", on: S.rec ? 0 : 1 });
document.addEventListener("pointerlockchange", () => { S.locked = document.pointerLockElement === canvas; if (S.locked) S.nolock = false; });
document.addEventListener("mousemove", e => { if (S.locked || S.drag) { inp.dx += e.movementX * S.sens; inp.dy -= e.movementY * S.sens; } });
function showMenu() { $("menu").classList.remove("hidden"); inp.keys.clear(); if (document.pointerLockElement) document.exitPointerLock(); }
function hideMenu(relock) { $("menu").classList.add("hidden"); if (relock) lock(); }
$("closeMenu").onclick = () => hideMenu(true);
$("newNothing").onclick = () => { reset("nothing"); hideMenu(true); };
$("newData").onclick = () => { reset("data"); hideMenu(true); };
function pause(on) { S.paused = on; send({ t: "pause", on: on ? 1 : 0 }); }
document.addEventListener("visibilitychange", () => { if (document.hidden && S.started && !S.paused) pause(true); });
addEventListener("blur", () => inp.keys.clear());

addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  if (["Tab", "Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Backquote"].includes(e.code)) e.preventDefault();
  if (!S.started) { if (e.code === "Enter") enter(); return; }
  if (e.repeat) return;
  const menu = !$("menu").classList.contains("hidden");
  if (e.code === "Tab" || e.code === "KeyL") { menu ? hideMenu(false) : showMenu(); return; }
  if (e.code.startsWith("Digit")) { const i = +e.code.slice(5); if (i < S.levels.length) { noclip(i); if (menu) hideMenu(false); } return; }
  if (menu) { if (e.code === "Escape") hideMenu(false); return; }
  inp.keys.add(e.code);
  switch (e.code) {
    case "KeyF": S.light ^= 1; audio.click(); toast(S.light ? "flashlight on" : "flashlight off", 900); break;
    case "KeyN": noclip(-1); break;
    case "KeyE": send({ t: "use" }); audio.click(); break;
    case "KeyR": reset(e.shiftKey || !S.hasData ? "nothing" : "data"); break;
    case "KeyH": S.hud = !S.hud; break;
    case "KeyV": S.vhs = !S.vhs; break;
    case "KeyM": S.muted = !S.muted; audio.mute(S.muted); toast(S.muted ? "sound off" : "sound on", 900); break;
    case "KeyP": pause(!S.paused); break;
    case "KeyO": send({ t: "rec", on: S.rec ? 0 : 1 }); break;
    case "Backquote": $("debug").classList.toggle("hidden"); if (!$("debug").classList.contains("hidden") && document.pointerLockElement) document.exitPointerLock(); break;
  }
});
addEventListener("keyup", e => inp.keys.delete(e.code));

// ---------------------------------------------------------------- settings panel
const fields = [["steps", "oSteps", 0], ["guide", "oGuide", 1], ["tau", "oTau", 2], ["shift", "oShift", 1], ["dread", "oDread", 2]];
function fillPanel() {
  for (const [id, out, dp] of fields) if (S.opts[id] !== undefined) { $(id).value = S.opts[id]; $(out).textContent = (+S.opts[id]).toFixed(dp); }
  $("exact").checked = !!S.opts.exact; $("sens").value = S.sens; $("oSens").textContent = S.sens.toFixed(4);
}
let setH = 0;
function pushOpts() {
  const o = { t: "set", exact: $("exact").checked ? 1 : 0 };
  for (const [id, out, dp] of fields) { o[id] = +$(id).value; $(out).textContent = o[id].toFixed(dp); }
  clearTimeout(setH); setH = setTimeout(() => send(o), 120);
}
for (const [id] of fields) $(id).addEventListener("input", pushOpts);
$("exact").addEventListener("change", pushOpts);
$("sens").addEventListener("input", () => { S.sens = +$("sens").value; $("oSens").textContent = S.sens.toFixed(4); });
$("invY").addEventListener("change", () => { S.invY = $("invY").checked; });
$("invX").addEventListener("change", () => { S.invX = $("invX").checked; });
$("blend").addEventListener("change", () => { S.blend = $("blend").checked; });

connect();
})();
