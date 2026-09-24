-- Release cells, but only the ones this owner still holds.
--
-- KEYS[1]     holds:owner:{owner}
-- KEYS[2..n]  hold:{provider}:{cell}
-- ARGV[1]     owner
--
-- Returns how many cells were deleted.
--
-- Why compare-and-delete, in one script: my hold can expire and someone else
-- can take the cell before I get round to releasing it. A plain DEL would then
-- delete *their* hold. GET-then-DEL from the client has the same bug with a
-- smaller window, because another client can run between the two commands.
local owner = ARGV[1]
local released = 0

for i = 2, #KEYS do
  if redis.call('GET', KEYS[i]) == owner then
    redis.call('DEL', KEYS[i])
    released = released + 1
  end
  redis.call('SREM', KEYS[1], KEYS[i])
end
return released
