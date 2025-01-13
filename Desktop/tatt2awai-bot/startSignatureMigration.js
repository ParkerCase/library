// startSignatureMigration.js
const signatureMigration = require('./signatureMigration');
const logger = require('./logger');
const fs = require('fs').promises;
const path = require('path');

// Add this constant at the top
const DEFAULT_TIMEOUT = 30000; // 30 seconds

const TIMEOUTS = {
    DROPBOX_OPERATION: 120000,    // 2 minutes for Dropbox operations
    IMAGE_PROCESSING: 60000,      // 1 minute for image processing
    DEFAULT: 30000                // 30 seconds default
};


const CONFIG = {
  PATHS: {
    LOGS: path.join(__dirname, 'logs'),
    ERROR_LOG: path.join(__dirname, 'logs', 'migration-error.log'),
    DIAGNOSTIC_LOG: path.join(__dirname, 'logs', 'migration-diagnostic.log')
  },
  EXIT_CODES: {
    SUCCESS: 0,
    VALIDATION_FAILED: 1,
    MIGRATION_FAILED: 2,
    INITIALIZATION_FAILED: 3
  }
};

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection:', {
    reason: reason instanceof Error ? reason.stack : reason,
    promise
  });
  process.exit(CONFIG.EXIT_CODES.MIGRATION_FAILED);
});

process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', {
    error: error.stack || error.message
  });
  process.exit(CONFIG.EXIT_CODES.MIGRATION_FAILED);
});

async function checkSystemRequirements() {
    const os = require('os');
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const availableMemory = freeMemory + getSwapAvailable();
    
    const REQUIRED_MEMORY = 256 * 1024 * 1024; // 256MB minimum
    
    if (availableMemory < REQUIRED_MEMORY) {
        await optimizeMemory();
        const newAvailableMemory = os.freemem() + getSwapAvailable();
        if (newAvailableMemory < REQUIRED_MEMORY) {
            logger.error('System requirements check failed:', {
                error: `Insufficient memory. Required: ${REQUIRED_MEMORY / 1024 / 1024}MB, Available: ${(newAvailableMemory / 1024 / 1024).toFixed(2)}MB`
            });
            throw new Error(`Insufficient memory. Required: ${REQUIRED_MEMORY / 1024 / 1024}MB, Available: ${(newAvailableMemory / 1024 / 1024).toFixed(2)}MB`);
        }
        return { memoryConstrained: true, availableMemory: newAvailableMemory };
    }
    
    return { memoryConstrained: false, availableMemory };
}

function getSwapAvailable() {
    try {
        const exec = require('child_process').execSync;
        const swapInfo = exec('free -b').toString();
        const swapLine = swapInfo.split('\n').find(line => line.startsWith('Swap:'));
        if (swapLine) {
            const [, total, used] = swapLine.split(/\s+/).map(Number);
            return total - used;
        }
    } catch (error) {
        console.warn('Unable to check swap space:', error);
    }
    return 0;
}

async function optimizeMemory() {
    if (global.gc) {
        global.gc();
    }
    
    Object.keys(require.cache).forEach(key => {
        if (!key.includes('node_modules')) {
            delete require.cache[key];
        }
    });
    
    const sharp = require('sharp');
    sharp.cache(false);
    await sharp.cache({ items: 20 });
    
    if (global.imageProcessingCache) {
        global.imageProcessingCache.clear();
    }
}

async function createDirectories() {
  try {
    await fs.mkdir(CONFIG.PATHS.LOGS, { recursive: true });
    return true;
  } catch (error) {
    logger.error('Failed to create required directories:', {
      error: error.message
    });
    throw error;
  }
}

async function cleanup() {
  try {
    const tempDir = path.join(require('os').tmpdir(), 'signature-migration');
    await fs.rm(tempDir, { recursive: true, force: true });

    const logFiles = await fs.readdir(CONFIG.PATHS.LOGS);
    for (const file of logFiles) {
      if (file.endsWith('.log')) {
        const stats = await fs.stat(path.join(CONFIG.PATHS.LOGS, file));
        if (Date.now() - stats.mtime.getTime() > 7 * 24 * 60 * 60 * 1000) {
          await fs.rename(
            path.join(CONFIG.PATHS.LOGS, file),
            path.join(CONFIG.PATHS.LOGS, 'archive', `${file}.${Date.now()}`)
          );
        }
      }
    }
  } catch (error) {
    logger.warn('Cleanup encountered issues:', {
      error: error.message
    });
  }
}


async function startMigration() {
    const startTime = Date.now();
    let memoryMonitor;
    
 try {
        logger.info('Starting migration process', {
            nodeVersion: process.version,
            platform: process.platform,
            memoryUsage: process.memoryUsage()
        });

        const memoryStatus = await checkSystemRequirements();
        
        if (memoryStatus.memoryConstrained) {
            logger.warn('Running in memory-constrained mode', {
                availableMemory: `${(memoryStatus.availableMemory / 1024 / 1024).toFixed(2)}MB`
            });
        }
        
        const migrationOptions = {
            batchSize: memoryStatus.memoryConstrained ? 1 : 5,
            memoryConstrained: memoryStatus.memoryConstrained,
            maxRetries: 3,
            timeouts: {
                dropbox: TIMEOUTS.DROPBOX_OPERATION,
                processing: TIMEOUTS.IMAGE_PROCESSING,
                default: TIMEOUTS.DEFAULT
            },
            processingOptions: {
                limitInputPixels: memoryStatus.memoryConstrained ? 
                    256 * 256 * 1024 : 512 * 512 * 1024,
                maxProcessingTime: TIMEOUTS.IMAGE_PROCESSING,
                optimizeMemory: true
            }
        };


        const migration = new signatureMigration(migrationOptions);
        
memoryMonitor = setInterval(async () => {
            const usage = process.memoryUsage();
            logger.debug('Memory usage:', {
                heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)}MB`,
                heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)}MB`,
                rss: `${Math.round(usage.rss / 1024 / 1024)}MB`
            });
            if (usage.heapUsed / usage.heapTotal > 0.8) {  // Lower threshold
                logger.info('Running memory optimization...');
                await optimizeMemory();
            }
        }, 2000);  // Check more frequently


        try {
            await migration.validate();
            const results = await migration.execute();
            
            const duration = ((Date.now() - startTime) / 60000).toFixed(2);
            logger.info('Migration completed', {
                duration: `${duration} minutes`,
                processed: results.success,
                failed: results.failed
            });

            return results;
        } finally {
            if (memoryMonitor) {
                clearInterval(memoryMonitor);
            }
        }
        
    } catch (error) {
        const duration = ((Date.now() - startTime) / 60000).toFixed(2);
        logger.error('Migration process failed:', {
            phase: error.phase || 'unknown',
            error: error.stack || error.message,
            duration: `${duration} minutes`
        });
        throw error;
    }
}

module.exports = {
  startMigration,
  checkSystemRequirements,
  cleanup
};
