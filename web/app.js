const $ = (id) => document.getElementById(id);
const ACTIONS = { 0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT" };
const state = { episode: null, frame: 0, timer: null, playing: false, pipTurn: null };

const INSTRUCTION_PHRASES = [
  ["leave the bathroom, and bedroom and go into the hall", "离开浴室和卧室，进入走廊"],
  ["enter the bedroom across from the starting room", "进入起始房间对面的卧室"],
  ["across from the starting room", "起始房间对面"],
  ["stop in the doorway to this bedroom", "在这间卧室的门口停下"],
  ["stop in the doorway", "在门口停下"],
  ["walk straight down the hall", "沿走廊直行"],
  ["walk straight down the hallway", "沿走廊直行"],
  ["walk down the hall", "沿走廊往前走"],
  ["go into the hall", "进入走廊"],
  ["enter the bedroom", "进入卧室"],
  ["leave the bathroom", "离开浴室"],
  ["leave the bedroom", "离开卧室"],
  ["and bedroom", "和卧室"],
  ["starting room", "起始房间"],
  ["living room", "客厅"],
  ["dining room", "餐厅"],
  ["dining table", "餐桌"],
  ["wait under", "在下方停下"],
  ["walk past", "走过"],
  ["walk through", "穿过"],
  ["turn left", "左转"],
  ["turn right", "右转"],
  ["on the left", "在左侧"],
  ["on the right", "在右侧"],
  ["to the left", "向左"],
  ["to the right", "向右"],
  ["doorway", "门口"],
  ["bathroom", "浴室"],
  ["bedroom", "卧室"],
  ["kitchen", "厨房"],
  ["hallway", "走廊"],
  ["stairs", "楼梯"],
  ["railing", "栏杆"],
  ["mirror", "镜子"],
  ["window", "窗户"],
  ["table", "桌子"],
  ["chair", "椅子"],
  ["stop", "停下"],
  ["wait", "停下等待"],
  ["enter", "进入"],
  ["leave", "离开"],
  ["walk", "走"],
  ["go", "去"],
];

function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("show");
  window.setTimeout(() => $("toast").classList.remove("show"), 3500);
}

async function getJson(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function formatValue(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (value == null) return "—";
  return String(value);
}

function formatMeters(value) {
  return value == null || Number.isNaN(Number(value)) ? "—" : `${Number(value).toFixed(2)} m`;
}

function translateInstruction(text) {
  if (!text) return "";
  let translated = ` ${text} `;
  for (const [english, chinese] of INSTRUCTION_PHRASES) {
    translated = translated.replace(new RegExp(english.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig"), chinese);
  }
  return translated
    .replace(/\s+and\s+/ig, "并")
    .replace(/\s+/g, " ")
    .replace(/\s+([，。、])/g, "$1")
    .replace(/([，。、])\s+/g, "$1")
    .replace(/,\s*/g, "，")
    .replace(/\.\s*/g, "。")
    .trim();
}

function currentTurn(frame) {
  const turnId = frame?.turn ?? 0;
  return state.episode?.trace.find((item) => Number(item.turn) === Number(turnId)) || null;
}

function turnAsset(turnId, suffix) {
  return state.episode?.assets?.[`turn_${String(turnId).padStart(2, "0")}_${suffix}.png`];
}

function decisionImageUrl(turnId) {
  for (const suffix of [
    "waypoint", "waypoint_FRONT", "waypoint_LEFT", "waypoint_RIGHT",
    "rgb", "current_FRONT",
  ]) {
    const url = turnAsset(turnId, suffix);
    if (url) return { url, suffix };
  }
  return { url: "", suffix: "" };
}

function selectedCoordinate(turn) {
  const parsed = turn?.parsed?.coordinate;
  if (Array.isArray(parsed) && parsed.length >= 2) return parsed.map(Number);
  const match = String(turn?.raw_output || "").match(/<(?:target|frontiers_coord)>\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)/i);
  return match ? [Number(match[1]), Number(match[2])] : null;
}

function selectedWaypoint(turn) {
  const coordinate = selectedCoordinate(turn);
  const waypoints = turn?.model_input?.waypoints || [];
  if (!coordinate) return waypoints[0] || null;
  return waypoints.find((item) => {
    const [x, y] = item.pixel_xy || [];
    return Math.abs(x - coordinate[0]) < 2 && Math.abs(y - coordinate[1]) < 2;
  }) || waypoints[0] || null;
}

function setImage(image, empty, url) {
  if (url) {
    image.src = url;
    image.style.display = "block";
    empty.style.display = "none";
  } else {
    image.removeAttribute("src");
    image.style.display = "none";
    empty.style.display = "block";
  }
}

function drawMarker(ctx, x, y, color, size = 11) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 16, y);
  ctx.lineTo(x + 16, y);
  ctx.moveTo(x, y - 16);
  ctx.lineTo(x, y + 16);
  ctx.stroke();
  ctx.restore();
}

function renderDecisionPip(turn) {
  const turnId = Number(turn?.turn ?? 0);
  const canvas = $("decision-canvas");
  const empty = $("pip-empty");
  const { url, suffix } = decisionImageUrl(turnId);
  const coordinate = selectedCoordinate(turn);
  const waypoints = turn?.model_input?.waypoints || [];
  $("pip-label").textContent = suffix.includes("waypoint") ? "WAYPOINT" : (coordinate ? "TARGET" : "RGB");
  $("decision-caption").textContent = coordinate
    ? `选中 (${coordinate[0].toFixed(0)}, ${coordinate[1].toFixed(0)}) · ${waypoints.length} 个候选`
    : (waypoints.length ? `${waypoints.length} 个候选路点` : "本轮没有标注路点");

  if (!url) {
    canvas.style.display = "none";
    empty.style.display = "block";
    return;
  }

  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    if (!suffix.includes("waypoint")) {
      waypoints.forEach((waypoint) => {
        const [x, y] = waypoint.pixel_xy || [];
        if (x == null || y == null) return;
        ctx.fillStyle = "rgba(30, 225, 70, .92)";
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#04120c";
        ctx.font = "bold 13px ui-sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(waypoint.waypoint_id), x, y);
      });
      if (coordinate) drawMarker(ctx, coordinate[0], coordinate[1], "#ff4d57", 13);
    }
    canvas.style.display = "block";
    empty.style.display = "none";
  };
  image.onerror = () => {
    canvas.style.display = "none";
    empty.style.display = "block";
  };
  image.src = url;
}

function renderTurnDistances(activeTurn) {
  const turns = state.episode?.trace || [];
  const total = turns.reduce((sum, turn) => sum + Number(turn.execution?.travelled_m || 0), 0);
  $("distance-total").textContent = `合计行驶 ${formatMeters(total)}`;
  $("turn-distance-list").replaceChildren(...turns.map((turn) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `distance-card${Number(turn.turn) === Number(activeTurn) ? " active" : ""}`;
    const title = document.createElement("span");
    title.textContent = `TURN ${Number(turn.turn) + 1}`;
    const distance = document.createElement("strong");
    distance.textContent = formatMeters(turn.execution?.travelled_m);
    const extra = document.createElement("em");
    const waypoint = selectedWaypoint(turn);
    extra.textContent = waypoint?.distance_m != null
      ? `目标 ${formatMeters(waypoint.distance_m)}`
      : (turn.execution?.status || "—");
    card.append(title, distance, extra);
    card.addEventListener("click", () => {
      const index = state.episode.frames.findIndex((frame) => Number(frame.turn) === Number(turn.turn));
      if (index >= 0) {
        state.frame = index;
        renderFrame();
      }
    });
    return card;
  }));
}

function renderFrame() {
  const data = state.episode;
  if (!data) return;
  const count = data.frames.length;
  state.frame = Math.max(0, Math.min(state.frame, Math.max(0, count - 1)));
  const frame = data.frames[state.frame];
  const turn = currentTurn(frame);
  const action = frame ? (ACTIONS[frame.action] || `ACTION_${frame.action}`) : "—";
  const coordinate = selectedCoordinate(turn);
  const waypoint = selectedWaypoint(turn);

  setImage($("agent-view"), $("empty-view"), frame?.url);
  $("timeline").max = Math.max(0, count - 1);
  $("timeline").value = state.frame;
  $("frame-counter").textContent = count ? `${state.frame + 1} / ${count}` : "0 / 0";
  $("frame-title").textContent = frame ? `FRAME ${String(frame.frame_index).padStart(4, "0")}` : "无逐步帧";
  $("turn-overlay").textContent = frame ? `TURN ${frame.turn}` : "TURN —";
  $("action-overlay").textContent = action;
  $("turn-counter").textContent = `TURN ${(frame?.turn ?? 0) + 1} / ${data.trace.length || data.result.turns || 0}`;

  const macros = turn?.parsed?.macro_tokens || [];
  $("decision-action").textContent = action;
  $("macro-list").replaceChildren(...macros.map((macro) => {
    const chip = document.createElement("span");
    const magnitude = macro.requested_cm != null ? `${macro.requested_cm}cm` :
      macro.requested_deg != null ? `${macro.requested_deg}°` : "";
    chip.textContent = `${macro.type}${magnitude ? `(${magnitude})` : ""}`;
    return chip;
  }));
  $("exec-status").textContent = turn?.execution?.status || "—";
  $("turn-value").textContent = turn ? String(Number(turn.turn) + 1) : "—";
  $("travelled").textContent = formatMeters(turn?.execution?.travelled_m);
  $("target-distance").textContent = formatMeters(waypoint?.distance_m);
  $("turn-actions").textContent = turn?.execution?.actions?.length ?? turn?.parsed?.action_sequence?.length ?? "—";
  $("target-coord").textContent = coordinate
    ? `(${coordinate[0].toFixed(0)}, ${coordinate[1].toFixed(0)})`
    : "—";
  $("latency").textContent = turn?.latency_s != null ? `${Number(turn.latency_s).toFixed(2)}s` : "—";
  $("parse-status").textContent = turn?.parsed?.parse_success === true ? "PARSED" :
    turn?.parsed?.parse_success === false ? "PARSE FAILED" : "—";
  $("model-output").textContent = turn?.raw_output || "该轮没有保存模型输出。";
  if (state.pipTurn !== Number(turn?.turn ?? -1)) {
    state.pipTurn = Number(turn?.turn ?? -1);
    renderDecisionPip(turn);
  }
  renderTurnDistances(turn?.turn);
  updateMapOptions(frame?.turn ?? 0);
}

function updateMapOptions(turnId) {
  const assets = state.episode?.assets || {};
  const options = [];
  if (assets["top_map_gt_pred.png"]) options.push(["top_map_gt_pred.png", "完整路线"]);
  for (const [suffix, label] of [["global_map", "全局地图"], ["local_map", "局部地图"], ["depth", "深度图"], ["input", "模型输入"]]) {
    const name = `turn_${String(turnId).padStart(2, "0")}_${suffix}.png`;
    if (assets[name]) options.push([name, label]);
  }
  const previous = $("map-kind").value;
  $("map-kind").replaceChildren(...options.map(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
  if (options.some(([value]) => value === previous)) $("map-kind").value = previous;
  renderMap();
}

function renderMap() {
  const url = state.episode?.assets?.[$("map-kind").value];
  setImage($("map-view"), $("no-map"), url);
}

function stop() {
  window.clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  $("play").textContent = "▶";
}

function play() {
  if (!state.episode?.frames.length) return;
  if (state.playing) return stop();
  state.playing = true;
  $("play").textContent = "Ⅱ";
  state.timer = window.setInterval(() => {
    if (state.frame >= state.episode.frames.length - 1) {
      if ($("loop").checked) state.frame = 0;
      else return stop();
    } else {
      state.frame += 1;
    }
    renderFrame();
  }, Number($("speed").value));
}

async function loadEpisode(key) {
  stop();
  try {
    const data = await getJson(`/api/episode/${key.split("/").map(encodeURIComponent).join("/")}`);
    state.episode = data;
    state.frame = 0;
    state.pipTurn = null;
    const result = data.result;
    const instruction = result.instruction || "—";
    $("instruction").textContent = instruction;
    const translated = translateInstruction(result.instruction || "");
    $("instruction-zh").hidden = !translated || translated === instruction;
    $("instruction-zh").textContent = translated;
    $("episode-badge").textContent = `EP ${result.episode_id ?? "—"}`;
    const success = Number(result.metrics?.success) === 1;
    $("success-badge").textContent = success ? "SUCCESS" : "NOT SUCCESS";
    $("success-badge").className = success ? "success" : "failure";
    $("scene").textContent = String(result.scene_id || "—").split("/").at(-1);
    const metrics = { ...(result.metrics || {}), total_actions: result.total_actions, turns: result.turns };
    $("metrics-grid").replaceChildren(...Object.entries(metrics).map(([name, value]) => {
      const div = document.createElement("div");
      div.className = "metric";
      const label = document.createElement("span");
      label.textContent = name.replaceAll("_", " ");
      const strong = document.createElement("strong");
      strong.textContent = formatValue(value);
      div.append(label, strong);
      return div;
    }));
    renderFrame();
  } catch (error) {
    toast(`轨迹加载失败：${error.message}`);
  }
}

async function init() {
  try {
    const catalog = await getJson("/api/episodes");
    const select = $("episode");
    if (!catalog.episodes.length) {
      select.innerHTML = "<option>没有找到含逐步帧的 episode</option>";
      toast(`在 ${catalog.outputs_root} 下未找到可回放轨迹`);
      return;
    }
    select.replaceChildren(...catalog.episodes.map((episode) => {
      const option = document.createElement("option");
      option.value = episode.key;
      option.textContent = `${episode.run} · EP ${episode.episode_id} · ${episode.frames} 帧`;
      return option;
    }));
    const requested = new URLSearchParams(window.location.search).get("episode");
    const preferred = catalog.episodes.find((item) => item.key === requested)
      || catalog.episodes.find((item) => item.key === "pred_eval/episode_60")
      || catalog.episodes.find((item) => item.key.endsWith("/episode_60"))
      || catalog.episodes.find((item) => item.key.endsWith("/episode_23"))
      || catalog.episodes[0];
    select.value = preferred.key;
    await loadEpisode(preferred.key);
  } catch (error) {
    toast(`无法扫描轨迹：${error.message}`);
  }
}

$("episode").addEventListener("change", (event) => loadEpisode(event.target.value));
$("timeline").addEventListener("input", (event) => { state.frame = Number(event.target.value); renderFrame(); });
$("play").addEventListener("click", play);
$("previous").addEventListener("click", () => { state.frame -= 1; renderFrame(); });
$("next").addEventListener("click", () => { state.frame += 1; renderFrame(); });
$("speed").addEventListener("change", () => { if (state.playing) { stop(); play(); } });
$("map-kind").addEventListener("change", renderMap);
window.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); play(); }
  if (event.code === "ArrowLeft") { state.frame -= 1; renderFrame(); }
  if (event.code === "ArrowRight") { state.frame += 1; renderFrame(); }
});

init();
