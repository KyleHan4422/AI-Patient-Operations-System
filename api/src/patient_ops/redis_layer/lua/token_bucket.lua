-- Take one token from a bucket, refilling it for the time that has passed.
--
-- KEYS[1]  rl:{scope}:{id}   HASH: tokens, ts
-- ARGV[1]  capacity
-- ARGV[2]  refill rate, tokens per millisecond
-- ARGV[3]  now, in milliseconds (the caller's clock)
--
-- Returns {allowed (1/0), milliseconds until a token is available}.
--
-- Refill and take in one script: two workers reading "1 token left" and both
-- taking it is exactly the burst a limiter exists to stop.
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local tokens = tonumber(redis.call('HGET', key, 'tokens') or capacity)
local ts = tonumber(redis.call('HGET', key, 'ts') or now)

-- A clock that went backwards (another worker's clock is a little behind)
-- refills nothing, rather than a negative amount.
if now > ts then
  tokens = math.min(capacity, tokens + (now - ts) * rate)
  ts = now
end

local allowed = 0
local retry_ms = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry_ms = math.ceil((1 - tokens) / rate)
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', ts)
-- Once the bucket would be full again, the key is no different from no key.
redis.call('PEXPIRE', key, math.ceil(capacity / rate) + 1000)
return {allowed, retry_ms}
