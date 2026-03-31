-- Migration 013: Add phone column to person table for SMS integration

ALTER TABLE person ADD COLUMN phone VARCHAR(20) DEFAULT NULL;
CREATE INDEX idx_person_phone ON person (phone);

-- Seed Lee's number
UPDATE person SET phone = '+14159872422' WHERE full_name LIKE '%Lee%' LIMIT 1;
