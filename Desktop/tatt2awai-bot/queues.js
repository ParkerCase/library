const Queue = require('bull');
const fs = require('fs');
const path = require('path');
const logger = require('./logger');
const sharp = require('sharp');


// Constants
const DOWNLOADS_DIR = path.join(__dirname, 'uploads');
const CHUNK_SIZE = 5 * 1024 * 1024;
const MAX_CONCURRENT_JOBS = 3;
const MAX_QUEUE_SIZE = 50;
const MAX_RETRIES = 3;
const CLEANUP_LOCK_KEY = 'cleanup_lock';
const CLEANUP_LOCK_TTL = 60;
const STALLED_JOB_TIMEOUT = 5 * 60 * 1000;
const HEALTH_CHECK_INTERVAL = 5 * 60 * 1000;
const RETRY_DELAY = 1000;
const JOB_TIMEOUT = 300000; // 5 minutes

const RATE_LIMIT_SETTINGS = {
    MAX_RETRIES: 3,
    RETRY_DELAY: 5000,
    MAX_CONCURRENT: 10,
    MIN_REFRESH_INTERVAL: 60000
};

// Ensure uploads directory exists
if (!fs.existsSync(DOWNLOADS_DIR)) {
    fs.mkdirSync(DOWNLOADS_DIR, { recursive: true });
}

// Initialize the queue
const processFileQueue = new Queue('file-processing', {
    redis: {
        port: process.env.REDIS_PORT || 6379,
        host: process.env.REDIS_HOST || 'localhost',
        retryStrategy: times => Math.min(times * 50, 2000),
        maxRetriesPerRequest: 3
    },
    defaultJobOptions: {
        attempts: MAX_RETRIES,
        backoff: {
            type: 'exponential',
            delay: RETRY_DELAY
        },
        removeOnComplete: true,
        removeOnFail: true,
        timeout: 300000 // 5 minutes
    },
    settings: {
        lockDuration: 30000,
        stalledInterval: 30000,
        maxStalledCount: 2
    }
});

// Queue event handlers
processFileQueue.on('error', error => {
    logger.error('Queue error:', error);
});

processFileQueue.on('waiting', jobId => {
    logger.info('Job waiting:', { jobId });
});

processFileQueue.on('active', job => {
    logger.info('Job started:', { jobId: job.id });
});

processFileQueue.on('completed', (job, result) => {
    logger.info('Job completed:', { jobId: job.id, result });
});

processFileQueue.on('failed', async (job, error) => {
    const dropboxManager = require('./dropboxManager');
    
    logger.error('Job failed:', { 
        jobId: job.id, 
        error: error.message,
        attempt: job.attemptsMade,
        maxAttempts: job.opts.attempts
    });

    // Track failed jobs
    if (job.data?.entry?.path_lower) {
        await dropboxManager.trackFailedOperation(
            job.data.entry.path_lower,
            error
        );
    }
});

processFileQueue.on('stalled', async (job) => {
    const dropboxManager = require('./dropboxManager');
    
    logger.warn('Job stalled:', { 
        jobId: job.id,
        lastActive: job.timestamp
    });

    // Track stalled jobs
    if (job.data?.entry?.path_lower) {
        await dropboxManager.trackFailedOperation(
            job.data.entry.path_lower,
            new Error('Job stalled')
        );
    }
});


// Add the process handler
processFileQueue.process(MAX_CONCURRENT_JOBS, async (job) => {
    const { entry } = job.data;
    let tempFilePath = null;
    let retries = 0;
    let fileBuffer = null;

async function attemptOperation(operation, name) {
    const dropboxManager = require('./dropboxManager');
    let attempts = 0;
    const maxAttempts = MAX_RETRIES;

    while (attempts < maxAttempts) {
        try {
            // Ensure fresh auth
            await dropboxManager.ensureAuth();
            
            // Use the rate-limited request wrapper
            const result = await dropboxManager.makeRateLimitedRequest(async () => {
                return await operation();
            });

            logger.info(`Operation ${name} succeeded:`, {
                hasResult: !!result,
                hasResultObject: !!result?.result,
                path: entry?.path_lower
            });

            return result;

        } catch (error) {
            attempts++;
            
            logger.error(`Operation ${name} failed:`, {
                error: error.message,
                attempt: attempts,
                path: entry?.path_lower,
                status: error?.status
            });

            // Handle rate limits
if (error?.status === 429) {
    const retryAfter = parseInt(error.headers?.['retry-after'] || '5', 10);
    const delay = Math.min(
        retryAfter * 1000 * Math.pow(2, attempts - 1),
        RATE_LIMIT_SETTINGS.MAX_RETRIES * RATE_LIMIT_SETTINGS.RETRY_DELAY
    );
    
    await job.progress({ 
        phase: 'rate_limited',
        retries: attempts,
        maxRetries: maxAttempts,
        nextRetry: Date.now() + delay,
        rateLimitInfo: {
            retryAfter,
            calculatedDelay: delay
        }
    });
    
    await new Promise(resolve => setTimeout(resolve, delay));
    continue;
}
            // Handle auth errors
            if (error?.status === 401) {
                await dropboxManager.refreshAccessToken();
                continue;
            }

            // Track failed operations
            await dropboxManager.trackFailedOperation(entry.path_lower, error);

            if (attempts === maxAttempts) {
                throw error;
            }

            // Exponential backoff for other errors
            await new Promise(resolve => 
                setTimeout(resolve, RETRY_DELAY * Math.pow(2, attempts - 1))
            );
        }
    }
    throw new Error(`Max retries reached during ${name}`);
}

    try {
        await job.progress({ phase: 'starting', progress: 0 });

        if (!entry || !entry.path_lower) {
            throw new Error('Invalid entry data: missing path_lower');
        }

        if (entry['.tag'] === 'folder') {
            return { success: true, skipped: true, reason: 'is_folder' };
        }

        const { processFileForAssistant, SUPPORTED_EXTENSIONS } = require('./openaiClient');
        const dropboxManager = require('./dropboxManager');
        const imageProcessor = require('./imageProcessor');
        const knowledgeBase = require('./knowledgeBase');

        const fileExt = path.extname(entry.path_lower).toLowerCase().replace('.', '');
        if (!SUPPORTED_EXTENSIONS.includes(fileExt)) {
            return { success: true, skipped: true, reason: 'unsupported_file_type' };
        }

        // Download file
// Download file
const fileResponse = await attemptOperation(
    async () => {
        const response = await dropboxManager.downloadFile(entry.path_lower);
        if (!response?.result?.fileBinary) {
            throw new Error('Invalid file response from Dropbox');
        }
        
        // Validate file binary data
        let fileBinary;
        if (Buffer.isBuffer(response.result.fileBinary)) {
            fileBinary = response.result.fileBinary;
        } else if (response.result.fileBlob) {
            const arrayBuffer = await response.result.fileBlob.arrayBuffer();
            fileBinary = Buffer.from(arrayBuffer);
        } else if (typeof response.result === 'object' && response.result.fileBinary) {
            fileBinary = Buffer.from(response.result.fileBinary);
        } else {
            fileBinary = Buffer.from(response.result);
        }
        
        fileBuffer = fileBinary;
        return {
            ...response,
            result: {
                ...response.result,
                fileBinary
            }
        };
    },
    'download'
);


        const fileName = `temp_${Date.now()}_${path.basename(entry.path_lower)}`;
        tempFilePath = path.join(DOWNLOADS_DIR, fileName);

        await job.progress({ phase: 'processing', progress: 40 });
        
        fs.writeFileSync(tempFilePath, fileBuffer);

        let analysisResult = null;
if (imageProcessor.getFileType(entry.name) === 'image') {
    analysisResult = await imageProcessor.processImage({
        path: tempFilePath,
        buffer: fileResponse.result.fileBinary,
        metadata: {
            originalPath: entry.path_lower,
            name: entry.name,
            size: entry.size,
            modifiedDate: entry.server_modified
        }
    });

    if (analysisResult) {
        // Create a simplified version for the knowledge base
        const knowledgeDoc = {
            analysis: {
                labels: analysisResult.labels,
                objects: analysisResult.objects,
                colors: analysisResult.colors
            },
            metadata: {
                path: entry.path_lower,
fileName: entry.name,               
 processedAt: new Date().toISOString(),
                size: entry.size,
                name: entry.name
            }
        };

        await knowledgeBase.addDocument(
            job.id.toString(),
            JSON.stringify(knowledgeDoc)
        );
    }
}

        await job.progress({ phase: 'finalizing', progress: 80 });

        // Store in knowledge base
        await knowledgeBase.addDocument(job.id.toString(), JSON.stringify({
            analysis: analysisResult,
            metadata: {
                path: entry.path_lower,
fileName: entry.name,
                processedAt: new Date().toISOString(),
                jobId: job.id,
                name: entry.name,
                size: entry.size,
                hasAnalysis: !!analysisResult
            }
        }));

        if (analysisResult) {
            await processFileForAssistant(
                fileBuffer,
                {
                    originalName: entry.name,
                    path: entry.path_lower,
                    type: fileExt,
                    tempPath: tempFilePath,
                    analysis: analysisResult
                }
            );
        }

        return {
            success: true,
            path: entry.path_lower,
            type: fileExt,
            processingTime: Date.now() - job.timestamp,
            analysis: analysisResult
        };

    } catch (error) {
        logger.error('Processing error:', {
            error: error.message,
            jobId: job.id,
            path: entry?.path_lower,
            stack: error.stack
        });
 // Track failed job
    const dropboxManager = require('./dropboxManager');
    await dropboxManager.trackFailedOperation(entry?.path_lower, error);
    
    // Allow retry for specific errors
    if (error?.status === 401 || error?.status === 429) {
        throw error; // Will trigger retry
    }

    // Mark as permanently failed for other errors
    return {
        success: false,
        error: error.message,
        permanent: true
    };

    } finally {
        if (tempFilePath && fs.existsSync(tempFilePath)) {
            try {
                fs.unlinkSync(tempFilePath);
                logger.info('Cleaned up temp file:', { path: tempFilePath });
            } catch (error) {
                logger.warn('Failed to clean up temp file:', {
                    path: tempFilePath,
                    error: error.message
                });
            }
        }
        fileBuffer = null;
        if (global.gc) {
            global.gc();
        }
    }
});

// Utility functions - keeping all your existing utils
async function healthCheck() {
    try {
        const client = processFileQueue.client;
        await client.ping();
        const [active, waiting] = await Promise.all([
            processFileQueue.getActive(),
            processFileQueue.getWaiting()
        ]);
        return {
            healthy: true,
            active: active.length,
            waiting: waiting.length,
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        return {
            healthy: false,
            error: error.message,
            timestamp: new Date().toISOString()
        };
    }
}

async function removeStaleJobs() {
    try {
        const staleJobs = await processFileQueue.getJobs(['active', 'waiting']);
        for (const job of staleJobs) {
            if (Date.now() - job.timestamp > JOB_TIMEOUT) {
                await job.remove();
                logger.info('Removed stale job:', { jobId: job.id });
            }
        }
    } catch (error) {
        logger.error('Error removing stale jobs:', error);
    }
}
// Add before module.exports
const queueEnhancements = {
    priorityScheduling: {
        calculatePriority(job) {
            const factors = {
                age: this.calculateAgeFactor(job.timestamp),
                size: this.calculateSizeFactor(job.data.size),
                type: this.calculateTypePriority(job.data.type)
            };
            
            return Object.values(factors).reduce((sum, factor) => sum + factor, 0);
        },
        
        calculateAgeFactor(timestamp) {
            const age = Date.now() - timestamp;
            return Math.min(age / (24 * 60 * 60 * 1000), 1); // Max 1 day
        },

        calculateSizeFactor(size) {
            const MAX_SIZE = 10 * 1024 * 1024; // 10MB
            return 1 - Math.min(size / MAX_SIZE, 1);
        },

        calculateTypePriority(type) {
            const priorities = {
                'image': 1.0,
                'sequence': 0.8,
                'document': 0.6
            };
            return priorities[type] || 0.5;
        },
        
        async rebalanceQueue() {
            const jobs = await processFileQueue.getWaiting();
            for (const job of jobs) {
                const priority = this.calculatePriority(job);
                await job.changePriority(priority);
            }
        }
    }
};

// Add this line to enhance the queue
Object.assign(processFileQueue, queueEnhancements);

// Export everything needed
module.exports = {
    processFileQueue,
    addJob: async (entry) => {
        try {
            // Ensure Redis connection is alive
            await processFileQueue.client.ping();
            // Validate entry before adding to queue
            if (!entry?.entry?.path_lower) {
                throw new Error('Invalid entry: missing path_lower');
            }
            return await processFileQueue.add(entry);
        } catch (error) {
            logger.error('Error adding job to queue:', error);
            throw error;
        }
    },
    getActive: () => processFileQueue.getActive(),
    getWaiting: () => processFileQueue.getWaiting(),
    healthCheck,
    removeStaleJobs,
    isReady: async () => {
        try {
            await processFileQueue.client.ping();
            return true;
        } catch {
            return false;
        }
    }
};
