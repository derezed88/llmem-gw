-- Add source verification tracking columns
ALTER TABLE mymcp.samaritan_sources
  ADD COLUMN verified_at       DATETIME     DEFAULT NULL,
  ADD COLUMN verification_methods VARCHAR(256) DEFAULT NULL,
  ADD COLUMN doc_modified_at   DATETIME     DEFAULT NULL;

-- Index to speed up batch candidate queries
CREATE INDEX idx_sources_verified_at ON mymcp.samaritan_sources (source_type, verified_at);
