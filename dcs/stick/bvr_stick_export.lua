-- bvr_stick_export.lua  --  telemetry out, stick commands in, for the player's aircraft
-- ======================================================================================
-- Runs in DCS's Export environment (Saved Games\DCS\Scripts\Export.lua loads it;
-- dcs/stick/install_export.py adds that line). Not the mission environment of
-- bvr_bridge.lua: this one sees the player's own cockpit data (LoGet*) and can
-- set the control axes (LoSetCommand), which a mission script cannot.
--
--   DCS --UDP 15401--> stick_test.py      one JSON line per frame, ev = "tel"
--   DCS <--UDP 15402-- stick_test.py      one text line per command
--
-- Commands (text lines, fields split by spaces):
--   AXES <seq> <pitch> <roll> <rudder> <throttle>   each -1..1, or - to leave it
--   RELEASE                                         pitch, roll, rudder to 0, control off
--   PING <id>                                       answered with ev = "pong"
--   CMD <code> <value>                              one raw LoSetCommand (experiments)
--   INFO                                            sends ev = "hello" again
--   IND [first [last]]                              the text of the cockpit displays (HUD, MFDs, DED), read-only:
--                                                   one ev = "ind" per display (list_indication of each id), then ev = "ind_end"
--   CLICK <device> <command> <value>                a cockpit switch or HOTAS control, as the
--                                                   module's own key bindings do (answered with
--                                                   ev = "click"); numbers from f16c_defs.json
-- If no AXES line arrives for CFG.watchdog seconds, pitch, roll and rudder go to 0
-- and control switches off (the F-16's flight control holds 1 g and the bank).
--
-- Telemetry goes out once per rendered frame by default: DCS may run the activity
-- event only at frame boundaries, which would turn a 50 Hz request into 30 Hz at
-- 60 fps. stick_test.py rate shows which source gives the better rate in your DCS.
-- Nothing is sent while the model time stands still (pause).
--
-- Everything the DCS API returns is read inside pcall: a function this aircraft
-- does not have is reported once in "hello" (api = {name = false}) and skipped.
-- Override any CFG field by defining BVR_STICK_CFG = {...} before the dofile.

local CFG = {
  host = "127.0.0.1", tx_port = 15401, rx_port = 15402,
  source = "frame",      -- "frame": a line per rendered frame (the Export frame callback);
                         -- "event": on LuaExportActivityNextEvent, rate times a second
  max_rate = 100,        -- "frame": at most this many lines a second (a 240 fps screen needs no more)
  rate = 50,             -- "event": telemetry lines per second asked of DCS
  objects_every = 5,     -- world objects and radar every Nth frame
  payload_every = 50,    -- weapons every Nth frame
  max_objects = 12,
  watchdog = 0.5,        -- s without an AXES line before the axes are zeroed
  stats_every = 5.0,     -- s (real time) between ev = "stats" lines
  allow_raw = true,      -- accept CMD lines
  ind_every = 20,        -- every Nth frame look for the radar and HUD markers below (0: never)
  allow_click = true,    -- accept CLICK lines (some devices and commands are always refused, below)
  codes = nil,           -- {pitch=, roll=, rudder=, throttle=} to force the command numbers
  version = 1,
}
if type(BVR_STICK_CFG) == "table" then
  for k, v in pairs(BVR_STICK_CFG) do CFG[k] = v end
end

-- ── logging and LuaSocket ──────────────────────────────────────────────
local function logmsg(s)
  pcall(function() log.write("bvr_stick", log.INFO, tostring(s)) end)
end

local socket = BVR_STICK_SOCKET
if not socket then
  local ok, s = pcall(function()
    package.path  = package.path  .. ";.\\LuaSocket\\?.lua;" .. lfs.currentdir() .. "\\LuaSocket\\?.lua"
    package.cpath = package.cpath .. ";.\\LuaSocket\\?.dll;" .. lfs.currentdir() .. "\\LuaSocket\\?.dll"
    return require("socket")
  end)
  if ok then socket = s else logmsg("LuaSocket not available: " .. tostring(s)) end
end

local function now()
  if socket and socket.gettime then return socket.gettime() end
  return os.clock()
end

-- ── JSON (flat enough for telemetry) ───────────────────────────────────
local function enc(v, depth)
  depth = depth or 0
  local t = type(v)
  if t == "number" then
    if v ~= v or v == math.huge or v == -math.huge then return "null" end
    return string.format("%.15g", v)         -- 10 digits would cut an epoch time to whole seconds
  elseif t == "boolean" then return v and "true" or "false"
  elseif t == "string" then
    return '"' .. v:gsub('[%c"\\]', function(c)
      return string.format("\\u%04x", c:byte())
    end) .. '"'
  elseif t == "table" and depth < 6 then
    local n = #v
    local out = {}
    if n > 0 then
      for i = 1, n do out[i] = enc(v[i], depth + 1) end
      return "[" .. table.concat(out, ",") .. "]"
    end
    for k, x in pairs(v) do
      if type(k) == "string" or type(k) == "number" then
        out[#out + 1] = enc(tostring(k)) .. ":" .. enc(x, depth + 1)
      end
    end
    return "{" .. table.concat(out, ",") .. "}"
  end
  return "null"
end

-- A JSON-safe copy of a table of unknown shape (the radar and payload tables):
-- numbers, strings and booleans kept, nested tables to a depth, at most 40 entries.
local function plain(v, depth)
  depth = depth or 0
  if type(v) ~= "table" then
    if type(v) == "number" or type(v) == "string" or type(v) == "boolean" then return v end
    return nil
  end
  if depth >= 3 then return nil end
  local out, n = {}, 0
  for k, x in pairs(v) do
    n = n + 1
    if n > 40 then break end
    local px = plain(x, depth + 1)
    if px ~= nil and (type(k) == "string" or type(k) == "number") then out[k] = px end
  end
  return out
end

-- ── sockets ────────────────────────────────────────────────────────────
local tx, rx
local function open_sockets()
  if not socket then return end
  tx = socket.udp(); tx:settimeout(0)
  rx = socket.udp(); rx:settimeout(0)
  local ok, err = rx:setsockname("127.0.0.1", CFG.rx_port)
  if not ok then logmsg("cannot listen on " .. CFG.rx_port .. ": " .. tostring(err)) end
end

local function send(tbl)
  if not tx then return end
  pcall(function() tx:sendto(enc(tbl), CFG.host, CFG.tx_port) end)
end

-- ── the DCS API, always inside pcall ───────────────────────────────────
local missing = {}
local function try(name, ...)
  local f = _G[name]
  if type(f) ~= "function" then missing[name] = true; return nil end
  local ok, a, b, c = pcall(f, ...)
  if ok then return a, b, c end
  missing[name] = tostring(a)
  return nil
end

local API = {"LoGetModelTime", "LoGetSelfData", "LoGetTrueAirSpeed", "LoGetIndicatedAirSpeed",
  "LoGetMachNumber", "LoGetAngleOfAttack", "LoGetAngleOfSideSlip", "LoGetAccelerationUnits",
  "LoGetVerticalVelocity", "LoGetAltitudeAboveSeaLevel", "LoGetAltitudeAboveGroundLevel",
  "LoGetADIPitchBankYaw", "LoGetVectorVelocity", "LoGetAngularVelocity", "LoGetEngineInfo",
  "LoGetWorldObjects", "LoGetTargetInformation", "LoGetLockedTargetInformation",
  "LoGetPayloadInfo", "LoSetCommand"}

-- The axis command numbers: DCS's own table if it can be read, else 2001-2004.
local CODES = {pitch = 2001, roll = 2002, rudder = 2003, throttle = 2004}
local CODES_FROM = "default"
local SEP = package.config and package.config:sub(1, 1) or "\\"
local function resolve_codes()
  if CFG.codes then
    for k, v in pairs(CFG.codes) do CODES[k] = v end
    CODES_FROM = "config"; return
  end
  local names = {pitch = "iCommandPlanePitch", roll = "iCommandPlaneRoll",
                 rudder = "iCommandPlaneRudder", throttle = "iCommandPlaneThrustCommon"}
  local tried = {}
  local function from(path)
    local f, lerr = loadfile(path)
    if not f then error(tostring(lerr)) end
    local env = setmetatable({}, {__index = _G})
    setfenv(f, env)
    f()
    local found, got = 0, {}
    for k, n in pairs(names) do
      local v = rawget(env, n)
      if type(v) == "number" then got[k] = v; found = found + 1 end
    end
    return found, got
  end
  local base = nil
  pcall(function() base = lfs.currentdir() end)
  if base then
    -- DCS's cockpit scripts load it from LockOn_Options.common_script_path (Scripts\Aircrafts\_Common\Cockpit)
    for _, rel in ipairs({"Scripts" .. SEP .. "Aircrafts" .. SEP .. "_Common" .. SEP .. "Cockpit" .. SEP .. "command_defs.lua",
                          "Scripts" .. SEP .. "command_defs.lua",
                          ".." .. SEP .. "Scripts" .. SEP .. "command_defs.lua"}) do
      local path = base .. SEP .. rel
      local ok, found, got = pcall(from, path)
      if ok and found == 4 then
        for k, v in pairs(got) do CODES[k] = v end
        CODES_FROM = "command_defs.lua"; return
      end
      tried[#tried + 1] = path .. ": " .. (ok and (found .. "/4 names found") or tostring(found)):sub(1, 90)
    end
  else
    tried[1] = "lfs.currentdir() not available"
  end
  CODES_FROM = "default (" .. table.concat(tried, "; ") .. ")"
end

-- ── state ──────────────────────────────────────────────────────────────
local S = {
  n = 0, frames = 0, events = 0, controlling = false,
  ax = {pitch = 0, roll = 0, rudder = 0, throttle = nil, seq = 0},
  last_rx = 0, t0_real = nil,
  stat_real = nil, stat_frames = 0, stat_events = 0, stat_dt_max = 0, stat_fdt_max = 0,
  last_ev_real = nil, last_frame_real = nil, started = false, sent_throttle = nil,
  last_emit_mt = nil, last_emit_rt = nil, stat_tel = 0,
}

local function setcmd(code, value)
  local ok, err = pcall(LoSetCommand, code, value)
  if not ok then
    if not S.cmd_err then S.cmd_err = true; logmsg("LoSetCommand failed: " .. tostring(err)) end
  end
  return ok
end

local function apply_axes()
  if not S.controlling then return end
  setcmd(CODES.pitch, S.ax.pitch)
  setcmd(CODES.roll, S.ax.roll)
  setcmd(CODES.rudder, S.ax.rudder)
  if S.ax.throttle ~= nil then setcmd(CODES.throttle, S.ax.throttle) end
end

local function zero_axes(why)
  setcmd(CODES.pitch, 0); setcmd(CODES.roll, 0); setcmd(CODES.rudder, 0)
  S.ax.pitch, S.ax.roll, S.ax.rudder = 0, 0, 0
  if S.controlling then send({ev = "released", why = why, rt = now()}) end
  S.controlling = false
end

local function hello()
  local api = {}
  for _, n in ipairs(API) do api[n] = type(_G[n]) == "function" end
  send({ev = "hello", version = CFG.version, codes = CODES, codes_from = CODES_FROM,
        api = api, source = CFG.source, rate = CFG.rate, max_rate = CFG.max_rate, watchdog = CFG.watchdog, rx_port = CFG.rx_port,
        rt = now()})
end

-- What the displays draw says what the radar is doing (read from `ind` dumps in DCS, 13 October): the radar page
-- (an MFD showing FCR_Antenna_Azimuth_Tics_Root) draws FCR_TrackedTgtPlaceholder once a target is designated, and
-- its search cursor (Cursor_Root) and scan limits go when the radar changes to a single-target track.
-- The HUD draws HUD_AA_TD_box on a designated target in view, HUD_TLL (target locator line) when it is out of view.
local MARK = {fcr = "FCR_Antenna_Azimuth_Tics_Root", tracked = "FCR_TrackedTgtPlaceholder", cursor = "Cursor_Root",
              sam = "FCR_SAM_TargetCarest", limits = "Azimuth_Scan_Limits_Left", td = "HUD_AA_TD_box",
              tll = "HUD_TLL", threat = "ThreatSymbol1"}
if type(CFG.markers) == "table" then for k, v in pairs(CFG.markers) do MARK[k] = v end end
local function has(txt, key) return type(txt) == "string" and txt:find(key, 1, true) ~= nil end
local function scan_displays()
  local rd = {}
  local ok, hud = pcall(list_indication, 1)
  if ok then rd.td, rd.tll = has(hud, MARK.td), has(hud, MARK.tll) end
  for id = 2, 6 do                                           -- the MFDs: whichever shows the radar page
    local ok2, txt = pcall(list_indication, id)
    if ok2 and has(txt, MARK.fcr) then
      rd.fcr, rd.tracked, rd.cursor = id, has(txt, MARK.tracked), has(txt, MARK.cursor)
      rd.sam, rd.limits = has(txt, MARK.sam), has(txt, MARK.limits)
      break
    end
  end
  local ok3, rwr = pcall(list_indication, 17)
  if ok3 then rd.threat = has(rwr, MARK.threat) end
  return rd
end

-- CLICK refuses whole devices and single commands whatever the script sends: the electrical,
-- fuel, hydraulic, engine, gear and oxygen systems, the canopy and ejection (CPT_MECH), the flight
-- control interface and the countermeasures dispenser, and the F-16C's stores jettison and
-- alternate release (MMC 3003 emergency jettison, 3004 and 3018 ground jettison, 3005 ALT REL).
-- Device ids and command numbers are the F-16C module's own (dcs/stick/f16c_defs.json).
local CLICK_DENY_DEVICE = {[2] = true, [3] = true, [4] = true, [5] = true, [6] = true, [7] = true,
                           [8] = true, [10] = true, [32] = true}
local CLICK_DENY = {["19:3003"] = true, ["19:3004"] = true, ["19:3005"] = true, ["19:3018"] = true}

-- ── input ──────────────────────────────────────────────────────────────
local function num(s, lo, hi)
  local v = tonumber(s)
  if v == nil then return nil end
  if lo and v < lo then v = lo end
  if hi and v > hi then v = hi end
  return v
end

local function handle(line)
  local w = {}
  for tok in line:gmatch("%S+") do w[#w + 1] = tok end
  local cmd = w[1]
  if cmd == "AXES" then
    S.ax.seq = tonumber(w[2]) or S.ax.seq
    local p, r, y, t = num(w[3], -1, 1), num(w[4], -1, 1), num(w[5], -1, 1), num(w[6], -1, 1)
    if p ~= nil then S.ax.pitch = p end
    if r ~= nil then S.ax.roll = r end
    if y ~= nil then S.ax.rudder = y end
    if t ~= nil then S.ax.throttle = t end
    S.controlling = true
    S.last_rx = now()
    apply_axes()
  elseif cmd == "RELEASE" then
    zero_axes("release")
  elseif cmd == "PING" then
    send({ev = "pong", id = w[2], rt = now(), t = try("LoGetModelTime")})
  elseif cmd == "CMD" and CFG.allow_raw then
    local code, v = num(w[2]), num(w[3])
    if code and v then send({ev = "raw", code = code, value = v, ok = setcmd(code, v), rt = now()}) end
  elseif cmd == "CLICK" and CFG.allow_click then
    local dev, c, v = num(w[2]), num(w[3]), num(w[4])
    if dev and c and v then
      if CLICK_DENY_DEVICE[dev] or CLICK_DENY[string.format("%d:%d", dev, c)] then
        send({ev = "click", ok = false, err = "refused", dev = dev, cmd = c, value = v, rt = now()})
      else
        local ok, err = pcall(function() GetDevice(0):performClickableAction(dev, c, v) end)
        send({ev = "click", ok = ok, err = (not ok) and tostring(err) or nil, dev = dev, cmd = c, value = v, rt = now()})
      end
    end
  elseif cmd == "IND" then
    -- one message per display (the HUD and the MFD pages are several KB each), then ev = "ind_end"
    local lo = math.floor(num(w[2], 0, 500) or 0)
    local hi = math.floor(math.min(num(w[3], 0, 500) or 49, lo + 99))
    local count = 0
    for id = lo, hi do
      local ok, txt = pcall(list_indication, id)
      if ok and type(txt) == "string" and #txt > 0 then
        count = count + 1
        send({ev = "ind", id = id, len = #txt, text = txt:sub(1, 30000)})
      end
    end
    send({ev = "ind_end", count = count, rt = now(), t = try("LoGetModelTime")})
  elseif cmd == "INFO" then
    hello()
  end
end

local function poll()
  if not rx then return end
  for _ = 1, 40 do
    local ok, data = pcall(function() return rx:receive() end)
    if not ok or not data then break end
    local ok2, err = pcall(handle, data)
    if not ok2 then logmsg("bad command '" .. tostring(data) .. "': " .. tostring(err)) end
  end
  if S.controlling and now() - S.last_rx > CFG.watchdog then zero_axes("watchdog") end
end

-- ── output ─────────────────────────────────────────────────────────────
local function vec(v)
  if type(v) ~= "table" then return nil end
  return {x = v.x, y = v.y, z = v.z}
end

local function objects(me)
  local all = try("LoGetWorldObjects")
  if type(all) ~= "table" then return nil end
  local list = {}
  local px, pz = 0, 0
  if me and me.Position then px, pz = me.Position.x or 0, me.Position.z or 0 end
  for id, o in pairs(all) do
    if type(o) == "table" and o.Position and not (me and o.Name == me.Name and o.UnitName == me.UnitName
                                                  and o.GroupName == me.GroupName) then
      local dx, dz = (o.Position.x or 0) - px, (o.Position.z or 0) - pz
      local ty = o.Type or {}
      list[#list + 1] = {d = math.sqrt(dx * dx + dz * dz), o = {
        id = id, name = o.Name, unit = o.UnitName, coal = o.CoalitionID or o.Coalition,
        l1 = ty.level1, l2 = ty.level2, l3 = ty.level3, l4 = ty.level4,
        lat = o.LatLongAlt and o.LatLongAlt.Lat, lon = o.LatLongAlt and o.LatLongAlt.Long,
        alt = o.LatLongAlt and o.LatLongAlt.Alt, hdg = o.Heading, pitch = o.Pitch, bank = o.Bank,
        x = o.Position.x, y = o.Position.y, z = o.Position.z}}
    end
  end
  table.sort(list, function(a, b) return a.d < b.d end)
  local out = {}
  for i = 1, math.min(#list, CFG.max_objects) do out[i] = list[i].o end
  return out
end

local function emit()
  local r, mt = now(), try("LoGetModelTime")
  if mt and S.last_emit_mt and mt <= S.last_emit_mt + 1e-6 then                -- paused
    if not S.last_idle or r - S.last_idle >= 1.0 then                         -- say so, once a second
      S.last_idle = r
      send({ev = "idle", rt = r, t = mt, why = "model time is not advancing (paused)"})
    end
    return
  end
  if CFG.source == "frame" and S.last_emit_rt and r - S.last_emit_rt < 0.8 / CFG.max_rate then return end
  S.last_emit_mt, S.last_emit_rt = mt, r
  S.n, S.stat_tel = S.n + 1, S.stat_tel + 1
  local f = {ev = "tel", n = S.n, rt = r, t = mt}
  local me = try("LoGetSelfData")
  if type(me) == "table" then
    local lla, pos, ty = me.LatLongAlt or {}, me.Position or {}, me.Type or {}
    f.me = {name = me.Name, coal = me.CoalitionID or me.Coalition, l1 = ty.level1, l2 = ty.level2,
            lat = lla.Lat, lon = lla.Long, alt = lla.Alt, hdg = me.Heading, pitch = me.Pitch,
            bank = me.Bank, x = pos.x, y = pos.y, z = pos.z}
  end
  f.tas, f.ias, f.mach = try("LoGetTrueAirSpeed"), try("LoGetIndicatedAirSpeed"), try("LoGetMachNumber")
  f.aoa, f.beta = try("LoGetAngleOfAttack"), try("LoGetAngleOfSideSlip")
  f.g = vec(try("LoGetAccelerationUnits"))
  f.vs, f.asl, f.agl = try("LoGetVerticalVelocity"), try("LoGetAltitudeAboveSeaLevel"),
                       try("LoGetAltitudeAboveGroundLevel")
  local ap, ab, ay = try("LoGetADIPitchBankYaw")
  if ap then f.adi = {pitch = ap, bank = ab, yaw = ay} end
  f.vel, f.w = vec(try("LoGetVectorVelocity")), vec(try("LoGetAngularVelocity"))
  f.eng = plain(try("LoGetEngineInfo"))
  f.ax = {pitch = S.ax.pitch, roll = S.ax.roll, rudder = S.ax.rudder, throttle = S.ax.throttle,
          seq = S.ax.seq, on = S.controlling}
  if S.n % CFG.objects_every == 0 then
    f.obj = objects(type(me) == "table" and me or nil)
    f.radar = {targets = plain(try("LoGetTargetInformation")),
               locked = plain(try("LoGetLockedTargetInformation"))}
  end
  if S.n % CFG.payload_every == 1 then f.pay = plain(try("LoGetPayloadInfo")) end
  if CFG.ind_every > 0 and S.n % CFG.ind_every == 0 then
    local okd, rd = pcall(scan_displays)
    if okd then f.rd = rd end
  end
  send(f)
end

local function stats()
  local r = now()
  S.stat_real = S.stat_real or r
  local dt = r - S.stat_real
  if dt < CFG.stats_every then return end
  local missing_now = {}
  for k, v in pairs(missing) do missing_now[k] = v end
  send({ev = "stats", rt = r, dt = dt, frames = S.stat_frames, events = S.stat_events,
        frame_hz = S.stat_frames / dt, event_hz = S.stat_events / dt, tel_hz = S.stat_tel / dt,
        frame_dt_max = S.stat_fdt_max, event_dt_max = S.stat_dt_max,
        model_t = try("LoGetModelTime"), missing = missing_now})
  S.stat_real, S.stat_frames, S.stat_events, S.stat_dt_max, S.stat_fdt_max, S.stat_tel = r, 0, 0, 0, 0, 0
end

-- ── the Export callbacks, chained to whatever Export.lua already had ───
local prev = {
  start = LuaExportStart, stop = LuaExportStop, before = LuaExportBeforeNextFrame,
  after = LuaExportAfterNextFrame, activity = LuaExportActivityNextEvent,
}
local prev_next, prev_dead = nil, (prev.activity == nil)

local function call_prev(f, ...)
  if f then local ok, err = pcall(f, ...); if not ok then logmsg("chained callback: " .. tostring(err)) end end
end

function LuaExportStart()
  call_prev(prev.start)
  pcall(function()
    resolve_codes()
    open_sockets()
    S.started = true
    hello()
    logmsg("started; axes " .. enc(CODES) .. " from " .. CODES_FROM)
  end)
end

function LuaExportStop()
  pcall(function()
    if S.controlling then zero_axes("stop") end
    send({ev = "bye", rt = now()})
    if tx then tx:close() end
    if rx then rx:close() end
    tx, rx, S.started = nil, nil, false
  end)
  call_prev(prev.stop)
end

function LuaExportBeforeNextFrame()
  call_prev(prev.before)
  if not S.started then return end
  local r = now()
  if S.last_frame_real then
    local d = r - S.last_frame_real
    if d > S.stat_fdt_max then S.stat_fdt_max = d end
  end
  S.last_frame_real = r
  S.frames, S.stat_frames = S.frames + 1, S.stat_frames + 1
  pcall(poll)
  pcall(apply_axes)
end

function LuaExportAfterNextFrame()
  call_prev(prev.after)
  if not S.started then return end
  if CFG.source == "frame" then
    local ok, err = pcall(emit)
    if not ok then logmsg("emit: " .. tostring(err)) end
    pcall(stats)
  end
end

function LuaExportActivityNextEvent(t)
  local mine = t + (CFG.source == "event" and 1.0 / CFG.rate or 1.0)
  if S.started then
    local r = now()
    if S.last_ev_real then
      local d = r - S.last_ev_real
      if d > S.stat_dt_max then S.stat_dt_max = d end
    end
    S.last_ev_real = r
    S.events, S.stat_events = S.events + 1, S.stat_events + 1
    pcall(poll)
    if CFG.source == "event" then
      local ok, err = pcall(emit)
      if not ok then logmsg("emit: " .. tostring(err)) end
      pcall(stats)
    end
    pcall(apply_axes)
  end
  local nxt = mine
  if not prev_dead and (prev_next == nil or t >= prev_next - 1e-6) then
    local ok, r = pcall(prev.activity, t)
    if ok and type(r) == "number" then prev_next = r else prev_dead = true end
  end
  if not prev_dead and prev_next and prev_next < nxt then nxt = prev_next end
  return nxt
end
