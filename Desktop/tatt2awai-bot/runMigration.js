#!/usr/bin/env node

const migration = require('./startSignatureMigration');
const logger = require('./logger');

async function main() {
    try {
        logger.info('Starting signature migration process');
        const result = await migration.startMigration();
        logger.info('Migration completed successfully', result);
        process.exit(0);
    } catch (error) {
        logger.error('Migration failed:', error);
        process.exit(1);
    }
}

main();
