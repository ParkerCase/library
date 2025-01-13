const logger = require('./logger');
const dropboxManager = require('./dropboxManager');
const { processFileQueue } = require('./queues');
const { syncContent } = require('./syncContent');
const imageSimilarity = require('./imageSimilarity');

async function initializeSmartSync() {
  try {
    await dropboxManager.ensureAuth();
    
    if (!processFileQueue) {
      throw new Error('Queue not initialized');
    }
    await processFileQueue.client.ping();
    
    await processFileQueue.clean(0, 'completed');
    await processFileQueue.clean(0, 'failed');
    
    setInterval(async () => {
      try {
        const health = await healthCheck();
        if (!health.healthy) {
          logger.error('Queue health check failed:', health);
          await reinitializeQueue();
        }
      } catch (error) {
        logger.error('Health check error:', error);
      }
    }, 5 * 60 * 1000);

    await syncContent({ metadataOnly: true });
    
    // Make image similarity available globally
    global.imageSimilarity = imageSimilarity;
    
    setInterval(async () => {
      try {
        await syncContent({ metadataOnly: true });
      } catch (error) {
        logger.error('Metadata refresh failed:', error);
      }
    }, 15 * 60 * 1000);

    return true;
  } catch (error) {
    logger.error('Smart sync initialization error:', error);
    return false;
  }
}

module.exports = { initializeSmartSync };
