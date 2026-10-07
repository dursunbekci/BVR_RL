--[[
bvr_logger.lua  --  record a DCS World air engagement for BVR_RL
================================================================

Writes one JSON object per line to Saved Games\DCS\Logs\bvr_rl_<n>.jsonl.
dcs_world.py reads that file and replays it through BVR_RL's observation
code, so the inputs a trained policy would see in DCS can be compared with
the ones it saw in training (dcs_obs_check.py).

Load it from the mission: a MISSION START trigger with the action
DO SCRIPT FILE -> bvr_logger.lua. It logs every airplane of both coalitions;
nothing in the mission has to be named in a particular way.

It needs the io and lfs modules, which DCS removes from mission scripts by
default. In <DCS install>\Scripts\MissionScripting.lua comment out these two
lines, and restore them when you are done (with them removed, any mission
you load can read and write files on your computer):

    sanitizeModule('io')
    sanitizeModule('lfs')

Lines written (all positions and velocities in DCS map axes, metres:
x north, y up, z east):

  {"format":"bvr_rl.dcs.v1", "rate":0.1, "t0":..., "theatre":"Caucasus"}
  {"t":12.3, "units":[{name, coal, type, x,y,z, vx,vy,vz,
                       fx,fy,fz (nose unit vector), ux,uy,uz (canopy-up
                       unit vector), fuel, radar_on, radar_tgt}],
             "weapons":[{id, type, shooter, target, x,y,z, vx,vy,vz}]}
  {"ev":"shot", "t":..., "id":..., "shooter":..., "target":..., "type":...}
  {"ev":"hit",  "t":..., "id":..., "shooter":..., "target":..., "type":...}
  {"ev":"weapon_gone", "t":..., "id":...}
  {"ev":"dead", "t":..., "unit":..., "cause":"dead|crash|ejection|..."}
  {"ev":"ammo", "t":..., "unit":..., "items":[{type, count, aam, guidance}]}

"t" is DCS mission time (timer.getTime()), in seconds.
Only air-to-air missiles are tracked as weapons; hits by anything else are
still logged as "hit" events, with the weapon's type.
]]

local CFG = {
  rate    = 0.1,      -- s between samples; BVR_RL's radar model runs at 10 Hz
  start   = 1.0,      -- s after this script runs before the first sample
  flush   = 10,       -- samples between file flushes
}

if not io or not lfs then
  trigger.action.outText("bvr_logger: io/lfs are sanitized. Edit Scripts\\MissionScripting.lua "
    .. "(see the top of bvr_logger.lua). Nothing is being recorded.", 30)
  return
end

-- ── minimal JSON encoder ─────────────────────────────────────────────
local enc
local function enc_num(v)
  if v ~= v or v == math.huge or v == -math.huge then return "null" end
  if v == math.floor(v) and math.abs(v) < 1e15 then return string.format("%d", v) end
  return string.format("%.4f", v)
end
local function enc_str(s)
  local out = s:gsub('[%c"\\]', function(c) return string.format("\\u%04x", c:byte()) end)
  return '"' .. out .. '"'
end
enc = function(v)
  local t = type(v)
  if t == "number" then return enc_num(v)
  elseif t == "string" then return enc_str(v)
  elseif t == "boolean" then return v and "true" or "false"
  elseif t == "table" then
    local n = #v
    if n > 0 or next(v) == nil then
      local parts = {}
      for i = 1, n do parts[i] = enc(v[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local parts = {}
    for k, val in pairs(v) do
      parts[#parts + 1] = enc_str(tostring(k)) .. ":" .. enc(val)
    end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return "null"
end

-- ── output file ──────────────────────────────────────────────────────
local dir = lfs.writedir() .. "Logs\\"
local fname = string.format("bvr_rl_%d_%d.jsonl", math.floor(timer.getAbsTime()), math.random(100000, 999999))
local f = io.open(dir .. fname, "w")
if not f then
  trigger.action.outText("bvr_logger: cannot open " .. dir .. fname, 30)
  return
end
local function write(obj) f:write(enc(obj), "\n") end

local theatre = "unknown"
pcall(function() theatre = env.mission.theatre end)
write({format = "bvr_rl.dcs.v1", rate = CFG.rate, t0 = timer.getTime(), theatre = theatre})

-- ── helpers ──────────────────────────────────────────────────────────
local function name_of(obj)
  if obj == nil then return nil end
  local ok, n = pcall(function() return obj:getName() end)
  if ok then return n end
  return nil
end

local function is_aam(weapon)
  local ok, d = pcall(function() return weapon:getDesc() end)
  return ok and d and d.category == Weapon.Category.MISSILE
         and d.missileCategory == Weapon.MissileCategory.AAM
end

local function log_ammo(unit)
  local ok, ammo = pcall(function() return unit:getAmmo() end)
  local items = {}
  if ok and ammo then
    for _, a in ipairs(ammo) do
      local d = a.desc or {}
      items[#items + 1] = {
        type = d.typeName or "?", count = a.count or 0,
        aam = (d.category == Weapon.Category.MISSILE and d.missileCategory == Weapon.MissileCategory.AAM),
        guidance = d.guidance or 0,
      }
    end
  end
  write({ev = "ammo", t = timer.getTime(), unit = unit:getName(), items = items})
end

-- ── state ────────────────────────────────────────────────────────────
local seen_units = {}      -- name -> true once its ammo has been logged
local weapons    = {}      -- key -> {obj, id, shooter, type}
local next_id    = 1
local n_samples  = 0

local function weapon_key(w)
  if w == nil then return nil end
  return tostring(w.id_)
end

-- ── events ───────────────────────────────────────────────────────────
local E = world.event
local DEAD_EVENTS = {}
local function dead_ev(id, cause) if id then DEAD_EVENTS[id] = cause end end
dead_ev(E.S_EVENT_DEAD, "dead")
dead_ev(E.S_EVENT_CRASH, "crash")
dead_ev(E.S_EVENT_EJECTION, "ejection")
dead_ev(E.S_EVENT_PILOT_DEAD, "pilot_dead")
dead_ev(E.S_EVENT_UNIT_LOST, "unit_lost")

local handler = {}
function handler:onEvent(e)
  local ok, err = pcall(function()
    local t = timer.getTime()
    if e.id == E.S_EVENT_SHOT and e.weapon and is_aam(e.weapon) then
      local id = next_id; next_id = next_id + 1
      local shooter = name_of(e.initiator)
      local tgt = nil
      pcall(function() tgt = name_of(e.weapon:getTarget()) end)
      local wtype = "?"
      pcall(function() wtype = e.weapon:getTypeName() end)
      weapons[weapon_key(e.weapon)] = {obj = e.weapon, id = id, shooter = shooter, type = wtype}
      write({ev = "shot", t = t, id = id, shooter = shooter, target = tgt, type = wtype})
      if e.initiator then log_ammo(e.initiator) end
    elseif e.id == E.S_EVENT_HIT then
      local w = e.weapon and weapons[weapon_key(e.weapon)]
      local wtype = "?"
      pcall(function() wtype = e.weapon:getTypeName() end)
      write({ev = "hit", t = t, id = w and w.id or -1, shooter = name_of(e.initiator),
             target = name_of(e.target), type = wtype})
    elseif DEAD_EVENTS[e.id] and e.initiator then
      write({ev = "dead", t = t, unit = name_of(e.initiator), cause = DEAD_EVENTS[e.id]})
    elseif e.id == E.S_EVENT_MISSION_END then
      f:flush(); f:close(); f = nil
    end
  end)
  if not ok then env.info("bvr_logger event error: " .. tostring(err)) end
end
world.addEventHandler(handler)

-- ── sampling ─────────────────────────────────────────────────────────
local function unit_entry(u)
  local p = u:getPosition()
  local v = u:getVelocity()
  local e = {
    name = u:getName(), coal = u:getCoalition(), type = u:getTypeName(),
    x = p.p.x, y = p.p.y, z = p.p.z,
    vx = v.x, vy = v.y, vz = v.z,
    fx = p.x.x, fy = p.x.y, fz = p.x.z,
    ux = p.y.x, uy = p.y.y, uz = p.y.z,
    fuel = u:getFuel(),
  }
  pcall(function()
    local on, tgt = u:getRadar()
    e.radar_on = on and true or false
    e.radar_tgt = name_of(tgt)
  end)
  return e
end

local function sample(_, now)
  if f == nil then return nil end
  local ok, err = pcall(function()
    local units = {}
    for _, side in ipairs({coalition.side.RED, coalition.side.BLUE}) do
      for _, g in ipairs(coalition.getGroups(side, Group.Category.AIRPLANE) or {}) do
        for _, u in ipairs(g:getUnits() or {}) do
          if u and u:isExist() and u:isActive() then
            units[#units + 1] = unit_entry(u)
            if not seen_units[u:getName()] then
              seen_units[u:getName()] = true
              log_ammo(u)
            end
          end
        end
      end
    end
    local wlist = {}
    for key, w in pairs(weapons) do
      if w.obj:isExist() then
        local p = w.obj:getPoint()
        local v = w.obj:getVelocity()
        local tgt = nil
        pcall(function() tgt = name_of(w.obj:getTarget()) end)
        wlist[#wlist + 1] = {id = w.id, type = w.type, shooter = w.shooter, target = tgt,
                             x = p.x, y = p.y, z = p.z, vx = v.x, vy = v.y, vz = v.z}
      else
        write({ev = "weapon_gone", t = now, id = w.id})
        weapons[key] = nil
      end
    end
    write({t = now, units = units, weapons = wlist})
    n_samples = n_samples + 1
    if n_samples % CFG.flush == 0 then f:flush() end
  end)
  if not ok then env.info("bvr_logger sample error: " .. tostring(err)) end
  return now + CFG.rate
end

timer.scheduleFunction(sample, nil, timer.getTime() + CFG.start)
trigger.action.outText("bvr_logger: recording to " .. dir .. fname, 10)
