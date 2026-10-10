-- Mock of the DCS mission scripting API for bvr_bridge.lua (Lua 5.1).
-- Blue follows its route task (turn 10 deg/s toward the far point, climb, speed);
-- an AttackUnit task with ROE OPEN_FIRE launches one missile after LAUNCH_DELAY.
T = 0
LAUNCH_DELAY = 1.5
local SCHED = {}
LOG = {}            -- controller calls: {who, what, arg}
timer = {getTime = function() return T end, getAbsTime = function() return 43200 + T end,
         scheduleFunction = function(f, a, t) SCHED[#SCHED + 1] = {f = f, a = a, t = t} end}
trigger = {action = {outText = function(s) LOG[#LOG + 1] = {"text", s} end}}
env = {info = function(s) LOG[#LOG + 1] = {"envinfo", s} end, mission = {theatre = "Caucasus"}}
Weapon = {Category = {SHELL = 0, MISSILE = 1, ROCKET = 2, BOMB = 3}, MissileCategory = {AAM = 1, SAM = 2}}
Group = {Category = {AIRPLANE = 0, HELICOPTER = 1}}
AI = {Option = {Air = {
  id = {NO_OPTION = -1, ROE = 0, REACTION_ON_THREAT = 1, RADAR_USING = 3, FLARE_USING = 4,
        RTB_ON_BINGO = 6, PROHIBIT_JETT = 15, PROHIBIT_AB = 16, MISSILE_ATTACK = 18},
  val = {ROE = {WEAPON_FREE = 0, OPEN_FIRE_WEAPON_FREE = 1, OPEN_FIRE = 2, RETURN_FIRE = 3, WEAPON_HOLD = 4},
         REACTION_ON_THREAT = {NO_REACTION = 0, PASSIVE_DEFENCE = 1, EVADE_FIRE = 2, BYPASS_AND_ESCAPE = 3, ALLOW_ABORT_MISSION = 4},
         RADAR_USING = {NEVER = 0, FOR_ATTACK_ONLY = 1, FOR_SEARCH_IF_REQUIRED = 2, FOR_CONTINUOUS_SEARCH = 3},
         MISSILE_ATTACK = {MAX_RANGE = 0, NEZ_RANGE = 1, HALF_WAY_RMAX_NEZ = 2, TARGET_THREAT_EST = 3, RANDOM_RANGE = 4}}}}}
local HANDLER
world = {event = {S_EVENT_SHOT = 1, S_EVENT_HIT = 2, S_EVENT_CRASH = 5, S_EVENT_EJECTION = 6,
                  S_EVENT_DEAD = 8, S_EVENT_PILOT_DEAD = 9, S_EVENT_MISSION_END = 12, S_EVENT_KILL = 28,
                  S_EVENT_UNIT_LOST = 30},
         addEventHandler = function(h) HANDLER = h end}

local function mkctrl(who)
  local c = {who = who, opts = {}, tasks = {}}
  function c:setOption(id, v) self.opts[id] = v; LOG[#LOG + 1] = {who, "opt", id, v} end
  function c:setTask(t) self.tasks = {t}; LOG[#LOG + 1] = {who, "setTask", t} end
  function c:pushTask(t) self.tasks[#self.tasks + 1] = t; LOG[#LOG + 1] = {who, "pushTask", t} end
  function c:popTask() self.tasks[#self.tasks] = nil; LOG[#LOG + 1] = {who, "popTask"} end
  function c:resetTask() self.tasks = {}; LOG[#LOG + 1] = {who, "resetTask"} end
  function c:setSpeed(v, keep) self.speed = v; LOG[#LOG + 1] = {who, "setSpeed", v} end
  return c
end

local nid = 100
local function mkunit(name, coal, x, z, hdg, spd, alt)
  nid = nid + 1
  local u = {name = name, coal = coal, x = x, z = z, y = alt, hdg = hdg, spd = spd,
             alive = true, ammo = 4, id_ = nid, uid = nid}
  u.ctrl = mkctrl(name)
  local g = {}
  function g:getController() return u.ctrl end
  function g:getUnits() return {u} end
  function u:getGroup() return g end
  function u:getID() return self.uid end
  function u:getName() return self.name end
  function u:getPlayerName() return nil end
  function u:getCoalition() return self.coal end
  function u:getTypeName() return "F-16C_50" end
  function u:isExist() return self.alive end
  function u:isActive() return true end
  function u:getFuel() return 0.7 end
  function u:getPoint() return {x = self.x, y = self.y, z = self.z} end
  function u:getVelocity() return {x = self.spd * math.cos(self.hdg), y = 0, z = self.spd * math.sin(self.hdg)} end
  function u:getPosition()
    return {p = {x = self.x, y = self.y, z = self.z},
            x = {x = math.cos(self.hdg), y = 0, z = math.sin(self.hdg)},
            y = {x = 0, y = 1, z = 0},
            z = {x = -math.sin(self.hdg), y = 0, z = math.cos(self.hdg)}}
  end
  function u:getRadar() return true, nil end
  function u:getAmmo()
    return {{count = self.ammo, desc = {typeName = "AIM_120C", category = 1, missileCategory = 1, guidance = 3}}}
  end
  return u
end
blue = mkunit("Viper-1", 2, -300000, 600000, 0.0, 250, 9000)
red  = mkunit("Bandit-1", 1, -220000, 601000, math.pi, 250, 9500)
local byname = {["Viper-1"] = blue, ["Bandit-1"] = red}
Unit = {getByName = function(n) return byname[n] end}
coalition = {side = {NEUTRAL = 0, RED = 1, BLUE = 2},
  getGroups = function(side, cat)
    local u = side == 1 and red or blue
    return {u:getGroup()}
  end}

-- A red missile aimed at blue (auto-defend tests): MOCK_RED_SHOT() fires it,
-- MOCK_RED_GONE() ends it.
function MOCK_RED_SHOT()
  local m = {x = red.x, y = red.y, z = red.z, alive = true, id_ = 888, v = {x = 0, y = 0, z = 0}}
  function m:isExist() return self.alive end
  function m:getDesc() return {category = 1, missileCategory = 1} end
  function m:getTypeName() return "AIM_120C" end
  function m:getTarget() return blue end
  function m:getPoint() return {x = self.x, y = self.y, z = self.z} end
  function m:getVelocity() return self.v end
  function m:destroy() self.alive = false end
  RED_MSL = m
  HANDLER:onEvent({id = world.event.S_EVENT_SHOT, initiator = red, weapon = m, time = T})
end
function MOCK_RED_GONE() if RED_MSL then RED_MSL.alive = false end end

local msl
local attack_since
local GUNS = 805306368
local function step_blue(dt)
  local c = blue.ctrl
  local top = c.tasks[#c.tasks]
  if top and top.id == "AttackUnit" and top.params.weaponType == GUNS then
    -- a guns-only attack from far away: a hard turn toward red (7 deg/s), no shot
    local want = math.atan2(red.z - blue.z, red.x - blue.x)
    local d = (want - blue.hdg + 3 * math.pi) % (2 * math.pi) - math.pi
    local mx = math.rad(7) * dt
    blue.hdg = blue.hdg + math.max(-mx, math.min(mx, d))
    attack_since = nil
    return
  end
  if top and top.id == "AttackUnit" and c.opts[0] == 2 then
    attack_since = attack_since or T
    if T - attack_since >= LAUNCH_DELAY and not msl and blue.ammo > 0 then
      msl = {x = blue.x, y = blue.y, z = blue.z, alive = true, id_ = 777, v = {x = 0, y = 0, z = 0}}
      function msl:isExist() return self.alive end
      function msl:getDesc() return {category = 1, missileCategory = 1} end
      function msl:getTypeName() return "AIM_120C" end
      function msl:getTarget() return red end
      function msl:getPoint() return {x = self.x, y = self.y, z = self.z} end
      function msl:getVelocity() return self.v end
      function msl:destroy() self.alive = false end
      blue.ammo = blue.ammo - 1
      HANDLER:onEvent({id = world.event.S_EVENT_SHOT, initiator = blue, weapon = msl, time = T})
    end
    return
  end
  attack_since = nil
  if top and top.id == "Mission" then
    local pts = top.params.route.points
    local p = pts[#pts]
    local want = math.atan2(p.y - blue.z, p.x - blue.x)
    local d = (want - blue.hdg + 3 * math.pi) % (2 * math.pi) - math.pi
    local mx = math.rad(10) * dt
    blue.hdg = blue.hdg + math.max(-mx, math.min(mx, d))
    blue.y = blue.y + math.max(-50 * dt, math.min(50 * dt, p.alt - blue.y))
    blue.spd = blue.spd + math.max(-10 * dt, math.min(10 * dt, p.speed - blue.spd))
  end
end

function RUN(dofile_path, until_t, on_tick)
  dofile(dofile_path)
  local dt = 0.05
  while T < until_t do
    T = T + dt
    step_blue(dt)
    for _, u in ipairs({blue, red}) do
      if u.alive then
        u.x = u.x + u.spd * math.cos(u.hdg) * dt
        u.z = u.z + u.spd * math.sin(u.hdg) * dt
      end
    end
    if msl and msl.alive then
      local dx, dy, dz = red.x - msl.x, red.y - msl.y, red.z - msl.z
      local r = math.sqrt(dx*dx + dy*dy + dz*dz)
      msl.v = {x = 900 * dx / r, y = 900 * dy / r, z = 900 * dz / r}
      msl.x, msl.y, msl.z = msl.x + msl.v.x * dt, msl.y + msl.v.y * dt, msl.z + msl.v.z * dt
      if r < 60 then
        HANDLER:onEvent({id = world.event.S_EVENT_HIT, initiator = blue, weapon = msl, target = red, time = T})
        msl.alive = false; red.alive = false
        HANDLER:onEvent({id = world.event.S_EVENT_KILL, initiator = blue, target = red, weapon = msl, time = T})
        HANDLER:onEvent({id = world.event.S_EVENT_DEAD, initiator = red, time = T + 30})
      end
    end
    local keep = {}
    for _, s in ipairs(SCHED) do
      if s.t <= T then
        local nxt = s.f(s.a, T)
        if nxt then keep[#keep + 1] = {f = s.f, a = s.a, t = nxt} end
      else keep[#keep + 1] = s end
    end
    SCHED = keep
    if on_tick then on_tick(T) end
  end
  HANDLER:onEvent({id = world.event.S_EVENT_MISSION_END, time = T})
end
