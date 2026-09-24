-- Settle an in-flight claim -- only if it is still ours.
--
-- KEYS[1]  idem:{idempotency key}
-- ARGV[1]  the value we claimed with: "inflight:{token}"
-- ARGV[2]  what to replace it with ("done:{ref}"), or "" to delete it
-- ARGV[3]  ttl in milliseconds for the replacement
--
-- Returns 1 if settled, 0 if the claim was no longer ours.
--
-- The token is what makes this safe. If the write took longer than the claim's
-- TTL, the claim expired and a second request may have claimed the key; a
-- plain SET or DEL here would overwrite or delete *its* claim.
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
if ARGV[2] == '' then
  redis.call('DEL', KEYS[1])
else
  redis.call('SET', KEYS[1], ARGV[2], 'PX', tonumber(ARGV[3]))
end
return 1
