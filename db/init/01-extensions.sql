-- Runs once, on first initialisation of the pgdata volume.
-- If the volume already exists this file is silently skipped -- use `make nuke`
-- to recreate from scratch.
CREATE EXTENSION IF NOT EXISTS vector;
