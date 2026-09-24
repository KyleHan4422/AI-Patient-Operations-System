-- Hold every cell of one slot for one owner -- all of them, or none.
--
-- KEYS[1]     holds:owner:{owner}   the owner's index, for "release all mine"
-- KEYS[2..n]  hold:{provider}:{cell} one key per grid cell the slot covers
-- ARGV[1]     owner (the conversation's thread id)
-- ARGV[2]     ttl in milliseconds
--
-- Returns 1 if held, 0 if any cell is held by someone else.
--
-- Check everything first, then write: nothing is written unless everything
-- can be, so a refused hold leaves no stray cells behind. Holding a cell the
-- owner already holds refreshes it -- offering the same slot twice is fine.
local owner = ARGV[1]
local ttl = tonumber(ARGV[2])

for i = 2, #KEYS do
  local current = redis.call('GET', KEYS[i])
  if current and current ~= owner then
    return 0
  end
end

for i = 2, #KEYS do
  redis.call('SET', KEYS[i], owner, 'PX', ttl)
  redis.call('SADD', KEYS[1], KEYS[i])
end
-- The index lives as long as the newest hold it lists. Older entries whose
-- keys have already expired are harmless: release compares before deleting.
redis.call('PEXPIRE', KEYS[1], ttl)
return 1
