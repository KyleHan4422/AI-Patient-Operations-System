-- One circuit-breaker transition, atomically. Mirrors breaker_step() in
-- redis_layer/breaker.py rule for rule; test_breaker.py runs one transition
-- table against both, so a change to one without the other fails the build.
--
-- KEYS[1]  breaker:{name}   HASH: state, failures, since_ms
-- ARGV[1]  event: allow | success | failure | abandon
-- ARGV[2]  now, in milliseconds (the caller's clock)
-- ARGV[3]  failure threshold
-- ARGV[4]  cooldown, in milliseconds
--
-- Returns {old state, new state, allowed (1/0)}.
--
-- Atomic is the point: when the cooldown ends, many workers ask "may I call?"
-- at once, and exactly one of them may become the probe.
local key = KEYS[1]
local event = ARGV[1]
local now = tonumber(ARGV[2])
local threshold = tonumber(ARGV[3])
local cooldown = tonumber(ARGV[4])

local state = redis.call('HGET', key, 'state') or 'closed'
local failures = tonumber(redis.call('HGET', key, 'failures') or '0')
local since = tonumber(redis.call('HGET', key, 'since_ms') or '0')
local old = state

local function save(new_state, new_failures, new_since)
  redis.call('HSET', key, 'state', new_state, 'failures', new_failures, 'since_ms', new_since)
  state = new_state
end

local allowed = 1

if event == 'allow' then
  if state ~= 'closed' then
    if now - since >= cooldown then
      save('half_open', 0, now)   -- this caller is the probe
    else
      allowed = 0
    end
  end
elseif event == 'abandon' then
  if state == 'half_open' then   -- a cancelled probe hands its turn back
    save('half_open', failures, now - cooldown)
  end
elseif event == 'success' then
  if state ~= 'open' then        -- a late success says nothing while open
    save('closed', 0, 0)
  end
elseif event == 'failure' then
  if state == 'closed' then
    failures = failures + 1
    if failures >= threshold then
      save('open', 0, now)
    else
      save('closed', failures, since)
    end
  elseif state == 'half_open' then
    save('open', 0, now)
  end
else
  return redis.error_reply('unknown breaker event ' .. tostring(event))
end

return {old, state, allowed}
