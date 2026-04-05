-- Tool execution stats: aggregate counts per (model, tool) pair
CREATE TABLE IF NOT EXISTS `{prefix}tool_stats` (
    `id`            INT(11)      NOT NULL AUTO_INCREMENT,
    `model`         VARCHAR(100) NOT NULL,
    `tool_name`     VARCHAR(100) NOT NULL,
    `call_count`    INT          NOT NULL DEFAULT 0,
    `success_count` INT          NOT NULL DEFAULT 0,
    `error_count`   INT          NOT NULL DEFAULT 0,
    `first_called`  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    `last_called`   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_model_tool` (`model`, `tool_name`),
    KEY `idx_last_called` (`last_called`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
