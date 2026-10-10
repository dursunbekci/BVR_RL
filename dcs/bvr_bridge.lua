--[[
bvr_bridge.lua  --  let a trained BVR_RL policy fly an aircraft in DCS World
=============================================================================

Mission script. Load it with a MISSION START trigger, action DO SCRIPT FILE ->
bvr_bridge.lua (dcs/make_mission.py builds a mission that does this).

What it does, ten times a second of mission time:
  * sends the state of every airplane and air-to-air missile to dcs_live.py
    over UDP, one JSON line per datagram, in bvr_logger.lua's format, plus
    the events (shots, hits, deaths, ammunition);
  * reads the commands dcs_live.py sends back and flies the agent aircraft
    on them.

The agent aircraft is an AI unit. A command is a heading, an altitude, a
speed and "fire or not"; the bridge turns heading, altitude and speed into a
two-waypoint route along the commanded heading (re-issued when the command
changes), so the DCS AI flies the aircraft the way the simulator's autopilot
does. To fire, it lets the AI shoot one active-radar missile at the red
aircraft (weapons are held otherwise) and takes control back after the
launch. While the policy is in control the AI's own reactions to threats are
off: the policy decides when to defend.

Until the first command arrives nothing is changed, so with dcs_live.py
--shadow (which never sends commands) the AI flies its own mission while the
policy only watches.

Commands (UDP datagrams to CFG.port_in, plain text):
  CMD <seq> <heading deg, map north, clockwise> <altitude m> <speed m/s> <fire 0|1>
  OPT eta <off|mission|abs>   locked arrival times on the route (see route_point)
  OPT near <m>                the route's first point this far ahead (0: no first point)
  OPT wpt <turn|flyover>      the route points' type (Turning Point or Fly Over Point)
  OPT redhold <0|1>           1: an AI red holds its fire (dcs/turn_test.py)
  STOP          give the aircraft back to the AI (end of an episode)
  DESTROY <id>  remove missile <id> (as numbered in the shot events): dcs_live.py's
                support rule says it has had no guidance for too long

Needs LuaSocket in the mission scripting environment: dcs/setup_dcs.py adds
it to <DCS install>\Scripts\MissionScripting.lua as the global bvr_rl_socket,
leaving io, lfs, os, require and package removed as DCS ships them.
]]

local CFG = {
  host        = "127.0.0.1",
  port_out    = 15301,      -- DCS -> dcs_live.py
  port_in     = 15302,      -- dcs_live.py -> DCS
  rate        = 0.1,        -- s between samples (BVR_RL's radar model runs at 10 Hz)
  start       = 1.0,        -- s after this script runs before the first sample
  agent       = nil,        -- unit name of the aircraft the policy flies; nil: the first blue airplane
  red         = nil,        -- unit name of its opponent; nil: the first red airplane
  red_attack  = true,       -- order an AI red aircraft to attack the agent at the start
  near_m      = 3000,       -- the route's first point, this far along the commanded heading
  far_m       = 60000,      -- and its second
  reissue_deg = 2.0,        -- re-issue the route when the heading command moves this much,
  reissue_s   = 10.0,       -- or this long after the last one
  fire_timeout = 8.0,       -- s to wait for the AI to launch before giving up
  -- Weapon.flag for the shot: AR_AAM (active radar air-to-air, the AIM-120).
  -- Set to nil to let the AI choose, if it never shoots with this set.
  fire_weapon = 134217728,
  resend_s    = 2.0,        -- header and ammunition repeated this often (late starters)
}

-- ── LuaSocket ────────────────────────────────────────────────────────
local socket = bvr_rl_socket
if not socket and require then pcall(function() socket = require("socket") end) end
if not socket then
  trigger.action.outText("bvr_bridge: LuaSocket is not available. Run dcs/setup_dcs.py "
    .. "(see dcs/README.md) and restart DCS.", 30)
  return
end
local tx = socket.udp()
tx:settimeout(0)
local rx = socket.udp()
rx:settimeout(0)
local ok_bind, bind_err = rx:setsockname(CFG.host, CFG.port_in)
if not ok_bind then
  trigger.action.outText("bvr_bridge: cannot listen on port " .. CFG.port_in .. ": "
    .. tostring(bind_err), 30)
  return
end

-- ── minimal JSON encoder (as in bvr_logger.lua) ──────────────────────
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
local function send(obj) tx:sendto(enc(obj), CFG.host, CFG.port_out) end

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

local function airplanes(side)
  local out = {}
  for _, g in ipairs(coalition.getGroups(side, Group.Category.AIRPLANE) or {}) do
    for _, u in ipairs(g:getUnits() or {}) do
      if u and u:isExist() and u:isActive() then out[#out + 1] = u end
    end
  end
  return out
end

local function find_unit(name, side)
  if name then
    local u = Unit.getByName(name)
    if u and u:isExist() then return u end
    return nil
  end
  return airplanes(side)[1]
end

local function is_player(u)
  local ok, p = pcall(function() return u:getPlayerName() end)
  return ok and p ~= nil
end

local function ammo_event(unit)
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
  send({ev = "ammo", t = timer.getTime(), unit = unit:getName(), items = items})
end

local function set_opt(ctrl, id, val)
  if ctrl and id ~= nil and val ~= nil then pcall(function() ctrl:setOption(id, val) end) end
end

-- ── who is who ───────────────────────────────────────────────────────
local agent = find_unit(CFG.agent, coalition.side.BLUE)
local red = find_unit(CFG.red, coalition.side.RED)
if not agent or not red then
  trigger.action.outText("bvr_bridge: need a blue and a red airplane (agent "
    .. tostring(CFG.agent or "first blue") .. ", red " .. tostring(CFG.red or "first red") .. ")", 30)
  return
end
local AGENT, RED = agent:getName(), red:getName()
local theatre = "unknown"
pcall(function() theatre = env.mission.theatre end)
local T0 = timer.getTime()
-- Sent in the header; dcs_live.py warns when a mission's bridge is older
-- than the options it was asked to use. 2: OPT eta. 3: OPT near, wpt, redhold.
-- 7: RTB_ON_BINGO off for both aircraft. (4-6, the DCS AI's own defence and
-- hot turns, exist on feature/envelope-target-alt only.)
local BRIDGE_VERSION = 7

local function header()
  send({format = "bvr_rl.dcs.v1", rate = CFG.rate, t0 = T0, theatre = theatre,
        bridge = BRIDGE_VERSION, agent = AGENT, red = RED})
end

local O = AI.Option.Air
local function ctrl_of(u)
  local ok, c = pcall(function() return u:getGroup():getController() end)
  if ok and c then return c end
  ok, c = pcall(function() return u:getController() end)
  return ok and c or nil
end

-- Red: an AI opponent is told to attack the agent.
if CFG.red_attack and not is_player(red) then
  local rc = ctrl_of(red)
  set_opt(rc, O.id.ROE, O.val.ROE.OPEN_FIRE_WEAPON_FREE)
  -- At bingo fuel (16%) the DCS AI flies home, ignoring its tasks: not in a test.
  set_opt(rc, O.id.RTB_ON_BINGO, false)
  pcall(function()
    rc:pushTask({id = "AttackUnit", params = {unitId = agent:getID(), groupAttack = false}})
  end)
end

-- ── control of the agent ─────────────────────────────────────────────
local S = {
  controlled = false,      -- true once a command has arrived
  cmd = nil,               -- the last command {seq, hdg, alt, spd, fire}
  issued = nil,            -- the command the current route was built from
  t_issued = -1e9,
  fire_pending = nil,      -- {seq, t} while waiting for the AI to launch
  last_fire_seq = -1,
  near_m = CFG.near_m,     -- OPT near
  wpt = "Turning Point",   -- OPT wpt
  redhold = false,         -- OPT redhold
}

local function take_control()
  local c = ctrl_of(agent)
  set_opt(c, O.id.ROE, O.val.ROE.WEAPON_HOLD)
  set_opt(c, O.id.REACTION_ON_THREAT, O.val.REACTION_ON_THREAT.NO_REACTION)
  set_opt(c, O.id.RADAR_USING, O.val.RADAR_USING.FOR_CONTINUOUS_SEARCH)
  set_opt(c, O.id.MISSILE_ATTACK, O.val.MISSILE_ATTACK.MAX_RANGE)
  set_opt(c, O.id.PROHIBIT_JETT, true)
  -- Let it use afterburner: in training the aircraft reaches the commanded
  -- speed with full thrust, but a DCS AI following a route stayed at ~280 m/s
  -- (Mach 0.9 at 10 km) when told 340.
  set_opt(c, O.id.PROHIBIT_AB, false)
  set_opt(c, O.id.RTB_ON_BINGO, false)
  S.controlled = true
  send({ev = "bridge", t = timer.getTime(), status = "control", agent = AGENT})
end

local function release_control()
  if not S.controlled then return end
  local c = ctrl_of(agent)
  set_opt(c, O.id.REACTION_ON_THREAT, O.val.REACTION_ON_THREAT.EVADE_FIRE)
  set_opt(c, O.id.ROE, O.val.ROE.OPEN_FIRE_WEAPON_FREE)
  pcall(function() c:resetTask() end)
  S.controlled, S.cmd, S.issued, S.fire_pending = false, nil, nil, nil
  send({ev = "bridge", t = timer.getTime(), status = "released", agent = AGENT})
end

-- eta: nil, or the time this point must be reached. A DCS AI on a route
-- will not light its afterburner for a speed, but does to make a locked
-- arrival time (OPT eta, from dcs_live.py --eta-lock); the time then rules
-- and the speed is not locked.
local function route_point(p, hdg, d, alt, spd, eta)
  return {
    type = "Turning Point", action = S.wpt,
    x = p.x + d * math.cos(hdg), y = p.z + d * math.sin(hdg),
    alt = alt, alt_type = "BARO", speed = spd, speed_locked = eta == nil,
    ETA = eta or 0, ETA_locked = eta ~= nil,
    task = {id = "ComboTask", params = {tasks = {}}},
  }
end

local function issue_route(cmd)
  local p = agent:getPoint()
  local hdg = math.rad(cmd.hdg)
  local eta1, eta2 = nil, nil
  if S.eta_clock then                 -- "mission": timer.getTime(); "abs": timer.getAbsTime()
    local now = (S.eta_clock == "abs") and timer.getAbsTime() or timer.getTime()
    eta1 = now + S.near_m / math.max(cmd.spd, 50)
    eta2 = now + CFG.far_m / math.max(cmd.spd, 50)
  end
  local points = {}
  if S.near_m > 0 then
    points[#points + 1] = route_point(p, hdg, S.near_m, cmd.alt, cmd.spd, eta1)
  end
  points[#points + 1] = route_point(p, hdg, CFG.far_m, cmd.alt, cmd.spd, eta2)
  local task = {id = "Mission", params = {airborne = true, route = {points = points}}}
  local ok, err = pcall(function() ctrl_of(agent):setTask(task) end)
  if not ok then env.info("bvr_bridge route error: " .. tostring(err)) end
  -- The route's speed alone did not get the commanded speed flown; order it too.
  pcall(function() ctrl_of(agent):setSpeed(cmd.spd, true) end)
  S.issued, S.t_issued = cmd, timer.getTime()
end

local function hdg_diff(a, b)
  return math.abs((a - b + 540) % 360 - 180)
end

local function fire_done(status)
  local c = ctrl_of(agent)
  set_opt(c, O.id.ROE, O.val.ROE.WEAPON_HOLD)
  pcall(function() c:popTask() end)
  send({ev = "fire", t = timer.getTime(), status = status, seq = S.fire_pending and S.fire_pending.seq})
  S.fire_pending = nil
  if S.cmd then issue_route(S.cmd) end
end

local function request_fire(seq)
  if S.fire_pending or not red:isExist() then return end
  local c = ctrl_of(agent)
  local params = {unitId = red:getID(), expend = "One", attackQtyLimit = true, attackQty = 1,
                  groupAttack = false}
  if CFG.fire_weapon then params.weaponType = CFG.fire_weapon end
  set_opt(c, O.id.ROE, O.val.ROE.OPEN_FIRE)
  local ok, err = pcall(function() c:pushTask({id = "AttackUnit", params = params}) end)
  S.fire_pending = {seq = seq, t = timer.getTime()}
  send({ev = "fire", t = timer.getTime(), status = ok and "requested" or "refused", seq = seq,
        reason = (not ok) and tostring(err) or nil})
  if not ok then fire_done("refused") end
end

local weapons = {}       -- key -> {obj, id, shooter, type}
local next_id = 1
local function weapon_key(w) return w and tostring(w.id_) or nil end

-- DESTROY <id>: dcs_live.py's support rule (as in training) says this missile
-- has had no guidance from its shooter for too long before its seeker took over.
local function destroy_missile(id)
  local status, shooter = "gone", nil
  for _, w in pairs(weapons) do
    if w.id == id then
      shooter = w.shooter
      if w.obj:isExist() then
        local ok = pcall(function() w.obj:destroy() end)
        status = ok and "destroyed" or "failed"
      end
      break
    end
  end
  send({ev = "support_lost", t = timer.getTime(), id = id, shooter = shooter, status = status})
end

local function on_command(msg)
  if msg:match("^STOP") then release_control(); return end
  local did = msg:match("^DESTROY%s+(%d+)")
  if did then destroy_missile(tonumber(did)); return end
  local near = msg:match("^OPT%s+near%s+([%d%.]+)")
  local wpt = msg:match("^OPT%s+wpt%s+(%a+)")
  local hold = msg:match("^OPT%s+redhold%s+([01])")
  if near or wpt or hold then
    local was = S.near_m .. S.wpt .. tostring(S.redhold)
    if near then S.near_m = math.min(tonumber(near), CFG.far_m / 2) end
    if wpt then S.wpt = (wpt == "flyover") and "Fly Over Point" or "Turning Point" end
    if hold then
      S.redhold = hold == "1"
      if red:isExist() and not is_player(red) then
        set_opt(ctrl_of(red), O.id.ROE, S.redhold and O.val.ROE.WEAPON_HOLD
                                         or O.val.ROE.OPEN_FIRE_WEAPON_FREE)
      end
    end
    if was ~= S.near_m .. S.wpt .. tostring(S.redhold) then
      S.issued = nil                                                     -- re-route now
    end
    -- Confirmed every time (not only on a change): UDP may drop the reply.
    send({ev = "bridge", t = timer.getTime(), status = "opt", near_m = S.near_m,
          wpt = (S.wpt == "Fly Over Point") and "flyover" or "turn", redhold = S.redhold})
    return
  end
  local clock = msg:match("^OPT%s+eta%s+(%a+)")
  if clock then
    local new = (clock == "mission" or clock == "abs") and clock or nil
    if new ~= S.eta_clock then
      S.eta_clock = new; S.issued = nil                                 -- re-route now
      send({ev = "bridge", t = timer.getTime(), status = "eta", clock = new or "off"})
    end
    return
  end
  local seq, hdg, alt, spd, fire = msg:match("^CMD%s+(%d+)%s+([%-%d%.eE]+)%s+([%-%d%.eE]+)%s+([%-%d%.eE]+)%s+(%d)")
  if not seq or not agent:isExist() then return end
  local cmd = {seq = tonumber(seq), hdg = tonumber(hdg) % 360, alt = tonumber(alt),
               spd = tonumber(spd), fire = tonumber(fire)}
  if not S.controlled then take_control() end
  S.cmd = cmd
  if cmd.fire == 1 and cmd.seq ~= S.last_fire_seq then
    S.last_fire_seq = cmd.seq
    request_fire(cmd.seq)
  end
  if S.fire_pending then return end          -- the attack task flies until the launch
  local i = S.issued
  if i == nil or hdg_diff(cmd.hdg, i.hdg) >= CFG.reissue_deg or math.abs(cmd.alt - i.alt) > 50
     or math.abs(cmd.spd - i.spd) > 1 or timer.getTime() - S.t_issued >= CFG.reissue_s then
    issue_route(cmd)
  end
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
      send({ev = "shot", t = t, id = id, shooter = shooter, target = tgt, type = wtype})
      if e.initiator then ammo_event(e.initiator) end
      if shooter == AGENT and S.fire_pending then fire_done("launched") end
    elseif e.id == E.S_EVENT_HIT then
      local w = e.weapon and weapons[weapon_key(e.weapon)]
      local wtype = "?"
      pcall(function() wtype = e.weapon:getTypeName() end)
      send({ev = "hit", t = t, id = w and w.id or -1, shooter = name_of(e.initiator),
            target = name_of(e.target), type = wtype})
    elseif E.S_EVENT_KILL and e.id == E.S_EVENT_KILL and e.target then
      -- the moment DCS decides a unit is destroyed, with killer and weapon
      local w = e.weapon and weapons[weapon_key(e.weapon)]
      send({ev = "kill", t = t, unit = name_of(e.target), killer = name_of(e.initiator),
            id = w and w.id or -1})
    elseif DEAD_EVENTS[e.id] and e.initiator then
      send({ev = "dead", t = t, unit = name_of(e.initiator), cause = DEAD_EVENTS[e.id]})
    elseif e.id == E.S_EVENT_MISSION_END then
      send({ev = "mission_end", t = t})
    end
  end)
  if not ok then env.info("bvr_bridge event error: " .. tostring(err)) end
end
world.addEventHandler(handler)

-- ── the 10 Hz loop ───────────────────────────────────────────────────
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

local last_resend = -1e9
local function tick(_, now)
  local ok, err = pcall(function()
    -- commands first, newest last
    while true do
      local msg = rx:receive()
      if not msg then break end
      on_command(msg)
    end
    if S.fire_pending and now - S.fire_pending.t > CFG.fire_timeout then fire_done("timeout") end

    if now - last_resend >= CFG.resend_s then
      last_resend = now
      header()
      for _, u in ipairs({agent, red}) do
        if u:isExist() then ammo_event(u) end
      end
    end

    local units = {}
    for _, side in ipairs({coalition.side.RED, coalition.side.BLUE}) do
      for _, u in ipairs(airplanes(side)) do units[#units + 1] = unit_entry(u) end
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
        send({ev = "weapon_gone", t = now, id = w.id})
        weapons[key] = nil
      end
    end
    send({t = now, units = units, weapons = wlist})
  end)
  if not ok then env.info("bvr_bridge tick error: " .. tostring(err)) end
  return now + CFG.rate
end

header()
timer.scheduleFunction(tick, nil, timer.getTime() + CFG.start)
trigger.action.outText(string.format("bvr_bridge: %s v %s, sending to %s:%d, listening on %d",
  AGENT, RED, CFG.host, CFG.port_out, CFG.port_in), 15)
