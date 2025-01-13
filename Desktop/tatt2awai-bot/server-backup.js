const express = require('express');
const multer = require('multer');
const path = require('path');
const http = require('http');
const https = require('https');
const sharp = require('sharp');
const fs = require('fs');
const cors = require('cors');
const logger = require('./logger');
const rateLimit = require('express-rate-limit');
const vision = require('@google-cloud/vision');
const { OpenAI } = require('openai');
const dropboxManager = require('./dropboxManager');
const knowledgeBase = require('./knowledgeBase');
const imageProcessor = require('./imageProcessor');
const { createClient } = require('@supabase/supabase-js');
const crypto = require('crypto');
require('dotenv').config();

console.log('Environment check:', {
    SUPABASE_URL: process.env.SUPABASE_URL,
    NODE_ENV: process.env.NODE_ENV
});

// Test DNS resolution
const dns = require('dns');
dns.resolve('rfnglcfyzoyqenofmsev.supabase.co', (err, addresses) => {
    if (err) {
        console.error('DNS Resolution Error:', err);
    } else {
        console.log('Resolved addresses:', addresses);
    }
});

const app = express();
const HEALTH_CHECK_INTERVAL = 60 * 60 * 1000; // 5 minutes
const HTTP_PORT = process.env.HTTP_PORT || 4000;
const HTTPS_PORT = process.env.HTTPS_PORT || 4001;
const MEMORY_CHECK_INTERVAL = 60000; // 1 minute
const TIMEOUT_MS = 10000; // 10 seconds timeout
const DROPBOX_TIMEOUT_MS = 15000; // 15 seconds for Dropbox operations
const DROPBOX_AUTH_CACHE_DURATION = 4 * 60 * 1000; // 4 minutes
let lastDropboxAuth = null;
let dropboxAuthToken = null;
const IMAGE_BATCH_SIZE = 500;
let allImageFiles = null;
let lastAuthTime = null;
const AUTH_TIMEOUT = 1000 * 60 * 5; // 5 minutes
const featureCache = new Map();
const AsyncLock = require('async-lock');
const lock = new AsyncLock();

const TATTOO_ANALYSIS_THRESHOLDS = {
    INK_DENSITY: {
        HIGH: 0.75,    // Original tattoo
        MEDIUM: 0.45,  // During treatment
        LOW: 0.25,     // Significant fading
        MINIMAL: 0.1   // Near complete removal
    },
    PATTERN_DETECTION: {
        DOT_CONFIDENCE: 0.7,
        PATTERN_DENSITY: 0.5,
        MINIMUM_DOTS: 10
    },
    FADING: {
        SIGNIFICANT: 0.4,
        MODERATE: 0.25,
        MINIMAL: 0.1
    },
    SEQUENCE_MATCHING: {
        VISUAL_SIMILARITY: 0.7,
        PATTERN_MATCH: 0.6,
        TIME_GAP_DAYS: 90
    }
};

// Add after your existing constants
const CACHE_SETTINGS = {
    AUTH_REFRESH_INTERVAL: 4 * 60 * 1000,
    BATCH_SIZE: 10,
    MAX_CONCURRENT_BATCHES: 3,
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY_BASE: 2000,
    BATCH_DELAY: 500,
    CACHE_TTL: 24 * 60 * 60 * 1000,
    SUPPORTED_FORMATS: ['.jpg', '.jpeg', '.png', '.webp'],
    MAX_FILE_SIZE: 10 * 1024 * 1024
};

const ANALYSIS_SETTINGS = {
    VISION_API: {
        MAX_LABELS: 20,
        MIN_CONFIDENCE: 0.7,
        FEATURES: [
            { type: 'LABEL_DETECTION', maxResults: 20 },
            { type: 'IMAGE_PROPERTIES' },
            { type: 'OBJECT_LOCALIZATION' },
            { type: 'TEXT_DETECTION' }
        ]
    },
    PROCESSING: {
        MIN_IMAGE_SIZE: 100,
        MAX_IMAGE_SIZE: 4096,
        QUALITY: 90
    }
};

let initializationProgress = {
    total: 0,
    processed: 0,
    status: 'not started',
    lastError: null
};

const imageMetadataCache = new Map();

const imageSignatureCache = new Map();
const cacheTTL = 1000 * 60 * 60 * 24; // 24 hours in milliseconds

const CACHE_VERSION = '1.0'; // Increment when analysis logic changes
const analysisCache = new Map();

const imageSignatures = new Map();
let isInitialized = false;

// First, let's enhance the ImageCacheManager with better progress tracking and resume capability

class ImageCacheManager {
    constructor() {
        this.imageSignatures = new Map();
        this.processedEntries = new Set();
        this.failedEntries = new Set();
        this.lastAuthTime = Date.now();
        this.isInitialized = false;
        this.authToken = null;
        this.AUTH_REFRESH_INTERVAL = 4 * 60 * 1000;
        this.initializationProgress = {
            total: 0,
            processed: 0,
            status: 'not started',
            lastError: null,
            startTime: null,
            endTime: null,
            resumePoint: null,
            processedPaths: new Set(),
            failedPaths: new Set(),
            currentBatch: null
        };
        this.processingLock = new AsyncLock();
    }

    async ensureAuth() {
        const now = Date.now();
        if (this.authToken && now - this.lastAuthTime < this.AUTH_REFRESH_INTERVAL) {
            return this.authToken;
        }

        try {
            const dropboxStatus = await dropboxManager.ensureAuth();
            if (!dropboxStatus) {
                throw new Error('Failed to authenticate with Dropbox');
            }
            this.authToken = dropboxStatus;
            this.lastAuthTime = now;
            return this.authToken;
        } catch (error) {
            logger.error('Authentication error:', error);
            throw error;
        }
    }

    async processBatch(batch, retryCount = 0) {
        const MAX_RETRIES = 3;
        const batchId = crypto.randomUUID();
        
        try {
            await this.ensureAuth();
            this.initializationProgress.currentBatch = {
                id: batchId,
                size: batch.length,
                processed: 0,
                startTime: Date.now()
            };

            const results = await Promise.all(batch.map(async (entry) => {
                if (this.processedEntries.has(entry.path_lower)) {
                    return null;
                }

                const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(entry.path_lower)}`);
                
                try {
                    // Enhanced file download with exponential backoff
                    const fileData = await this.downloadWithRetry(entry.path_lower);
                    if (!fileData?.result?.fileBinary) {
                        throw new Error('Failed to download file after retries');
                    }

                    fs.writeFileSync(tempPath, fileData.result.fileBinary);

                    // Enhanced image validation
                    const metadata = await this.validateImage(tempPath);
                    const stats = await this.getImageStats(tempPath);
                    const visionResult = await this.analyzeWithVision(fileData.result.fileBinary);

                    // Build comprehensive signature
                    const signature = await this.buildImageSignature(entry, metadata, stats, visionResult);

                    // Store with transaction
                    await this.storeSignature(entry.path_lower, signature);

                    this.processedEntries.add(entry.path_lower);
                    this.initializationProgress.processedPaths.add(entry.path_lower);
                    this.initializationProgress.currentBatch.processed++;

                    return signature;

                } catch (error) {
                    logger.error(`Error processing ${entry.path_lower}:`, error);
                    this.failedEntries.add(entry.path_lower);
                    this.initializationProgress.failedPaths.add(entry.path_lower);
                    throw error;
                } finally {
                    if (fs.existsSync(tempPath)) {
                        fs.unlinkSync(tempPath);
                    }
                }
            }));

            return results.filter(Boolean);

        } catch (error) {
            if (retryCount < MAX_RETRIES) {
                logger.warn(`Retrying batch ${batchId}, attempt ${retryCount + 1}`);
                await new Promise(resolve => setTimeout(resolve, Math.pow(2, retryCount) * 1000));
                return this.processBatch(batch, retryCount + 1);
            }
            throw error;
        }
    }

    async downloadWithRetry(path, attempt = 0) {
        const MAX_ATTEMPTS = 5;
        const BACKOFF_MS = 1000;

        try {
            return await dropboxManager.downloadFile(path);
        } catch (error) {
            if (attempt < MAX_ATTEMPTS && (error?.status === 503 || error?.status === 429)) {
                await new Promise(resolve => 
                    setTimeout(resolve, BACKOFF_MS * Math.pow(2, attempt))
                );
                return this.downloadWithRetry(path, attempt + 1);
            }
            throw error;
        }
    }

async function validateImage(imagePath) {
    try {
        const metadata = await sharp(imagePath).metadata();
        return {
            isValid: !!(metadata && metadata.width && metadata.height),
            metadata
        };
    } catch (error) {
        logger.warn(`Image validation failed for ${imagePath}:`, error);
        return {
            isValid: false,
            error: error.message
        };
    }
}

    async getImageStats(imagePath) {
        const stats = await sharp(imagePath).stats();
        if (!stats?.channels || stats.channels.length < 3) {
            throw new Error('Invalid image statistics');
        }
        return stats;
    }

    async analyzeWithVision(imageBuffer) {
        const result = await visionClient.annotateImage({
            image: { content: imageBuffer.toString('base64') },
            features: ANALYSIS_SETTINGS.VISION_API.FEATURES
        });
        return result[0];
    }

    async buildImageSignature(entry, metadata, stats, visionResult) {
        return {
            path: entry.path_lower,
            fileName: entry.name,
            visionAnalysis: {
                labels: visionResult.labelAnnotations || [],
                objects: visionResult.localizedObjectAnnotations || [],
                colors: visionResult.imagePropertiesAnnotation?.dominantColors?.colors || [],
                text: visionResult.textAnnotations || []
            },
            imageStats: {
                means: stats.channels.map(c => c.mean),
                stdDevs: stats.channels.map(c => c.std),
                histogram: stats.channels.map(c => c.histogram)
            },
            visualFeatures: {
                aspectRatio: metadata.width / metadata.height,
                orientation: metadata.orientation || 1,
                format: metadata.format,
                dimensions: {
                    width: metadata.width,
                    height: metadata.height
                }
            },
            metadata: {
                size: entry.size,
                modified: entry.server_modified,
                directory: path.dirname(entry.path_lower),
                hash: crypto.createHash('md5').update(entry.path_lower).digest('hex'),
                analyzedAt: new Date().toISOString()
            }
        };
    }

    async storeSignature(path, signature) {
        this.imageSignatures.set(path, signature);
        await supabase
            .from('image_signatures')
            .upsert({
                path,
                signature,
                analyzed_at: new Date().toISOString()
            });
    }

    async initialize() {
        try {
            this.initializationProgress.status = 'starting';
            this.initializationProgress.startTime = Date.now();
            logger.info('Starting complete image cache initialization...');

            // Initial authentication
            await this.ensureAuth();

            // Get all entries with pagination
            const entries = await this.getAllEntriesWithPagination();
            
            const imageFiles = entries.filter(entry => 
                CACHE_SETTINGS.SUPPORTED_FORMATS.some(ext => 
                    entry.path_lower.endsWith(ext)
                )
            );

            this.initializationProgress.total = imageFiles.length;
            logger.info(`Found ${imageFiles.length} images to process`);

            // Process in optimized batches with concurrency control
            const batchSize = CACHE_SETTINGS.BATCH_SIZE;
            const concurrentBatches = CACHE_SETTINGS.MAX_CONCURRENT_BATCHES;

            for (let i = 0; i < imageFiles.length; i += (batchSize * concurrentBatches)) {
                // Check for auth refresh
                await this.ensureAuth();

                const batchPromises = [];
                for (let j = 0; j < concurrentBatches; j++) {
                    const startIdx = i + (j * batchSize);
                    const batch = imageFiles.slice(startIdx, startIdx + batchSize);
                    if (batch.length > 0) {
                        batchPromises.push(this.processBatch(batch));
                    }
                }

                await Promise.all(batchPromises);
                
                // Update progress
                this.initializationProgress.processed = this.processedEntries.size;
                logger.info(`Processed ${this.processedEntries.size}/${imageFiles.length} images`);
                
                // Save progress for potential resume
                await this.saveProgress();
                
                // Add delay between large batch groups
                await new Promise(resolve => setTimeout(resolve, CACHE_SETTINGS.BATCH_DELAY));
            }

            this.isInitialized = true;
            this.initializationProgress.status = 'complete';
            this.initializationProgress.endTime = Date.now();

            await this.finalizeInitialization();

        } catch (error) {
            this.initializationProgress.status = 'error';
            this.initializationProgress.lastError = error.message;
            logger.error('Initialization error:', error);
            
            // Save state for potential resume
            await this.saveProgress();
            
            throw error;
        }
    }

    async getAllEntriesWithPagination(cursor = null, accumulated = []) {
        const limit = 2000; // Dropbox API pagination limit
        const result = await dropboxManager.fetchDropboxEntries('', cursor, limit);
        
        if (!result?.result?.entries) {
            throw new Error('Failed to fetch entries from Dropbox');
        }

        accumulated.push(...result.result.entries);

        if (result.result.has_more && result.result.cursor) {
            return this.getAllEntriesWithPagination(result.result.cursor, accumulated);
        }

        return accumulated;
    }

    async saveProgress() {
        const progress = {
            timestamp: Date.now(),
            processed: Array.from(this.processedEntries),
            failed: Array.from(this.failedEntries),
            lastBatch: this.initializationProgress.currentBatch
        };

        await supabase
            .from('initialization_progress')
            .upsert({
                id: 'latest',
                progress,
                updated_at: new Date().toISOString()
            });
    }

    async finalizeInitialization() {
        const duration = this.initializationProgress.endTime - this.initializationProgress.startTime;
        
        logger.info('Cache initialization complete:', {
            total: this.initializationProgress.total,
            processed: this.processedEntries.size,
            failed: this.failedEntries.size,
            duration: `${Math.round(duration / 1000)}s`,
            averageProcessingTime: `${Math.round(duration / this.processedEntries.size)}ms per image`
        });

        // Clear progress data
        await supabase
            .from('initialization_progress')
            .delete()
            .eq('id', 'latest');
    }
}

// Enhanced sequence detector with more sophisticated pattern recognition
class SequenceDetector {
    constructor() {
        this.timeThreshold = 90 * 24 * 60 * 60 * 1000; // 90 days in ms
        this.similarityThreshold = 0.7;
    }

    async findRelatedImages(imagePath, imageSignatures) {
        const currentImage = imageSignatures.get(imagePath);
        if (!currentImage) return [];

        const directory = path.dirname(imagePath);
        const dirImages = Array.from(imageSignatures.values())
            .filter(img => path.dirname(img.path) === directory);

        // Sort by timestamp
        dirImages.sort((a, b) => 
            new Date(a.metadata.modified) - new Date(b.metadata.modified)
        );

        // Enhanced sequence detection
        const sequences = this.detectSequences(dirImages);
        
        // Find the sequence containing our target image
        const targetSequence = sequences.find(seq => 
            seq.images.some(img => img.path === imagePath)
        );

        if (!targetSequence) return [];

        // Enhance sequence with progression analysis
        return this.enhanceSequence(targetSequence);
    }

    detectSequences(images) {
        const sequences = [];
        let currentSequence = [];
        
        for (let i = 0; i < images.length; i++) {
            const current = images[i];
            
            if (currentSequence.length === 0) {
                currentSequence.push(current);
                continue;
            }

            const lastImage = currentSequence[currentSequence.length - 1];
            const timeDiff = new Date(current.metadata.modified) - new Date(lastImage.metadata.modified);
            const similarity = this.calculateImageSimilarity(current, lastImage);

            if (timeDiff <= this.timeThreshold && similarity >= this.similarityThreshold) {
                currentSequence.push(current);
            } else {
                if (currentSequence.length > 1) {
                    sequences.push({
                        images: [...currentSequence],
                        timespan: {
                            start: currentSequence[0].metadata.modified,
                            end: currentSequence[currentSequence.length - 1].metadata.modified
                        }
                    });
                }
                currentSequence = [current];
            }
        }

        // Don't forget the last sequence
        if (currentSequence.length > 1) {
            sequences.push({
                images: [...currentSequence],
                timespan: {
                    start: currentSequence[0].metadata.modified,
                    end: currentSequence[currentSequence.length - 1].metadata.modified
                }
            });
        }

        return sequences;
    }

    enhanceSequence(sequence) {
        const enhancedImages = sequence.images.map((img, index) => ({
            ...img,
            sequence: {
                position: index + 1,
                total: sequence.images.length,
                timeSincePrevious: index > 0 ? 
                    new Date(img.metadata.modified) - new Date(sequence.images[index - 1].metadata.modified) : 0,
                progress: this.calculateProgress(sequence.images[0], img)
            }
        }));

        return {
            images: enhancedImages,
            analysis: this.analyzeSequence(enhancedImages),
            timespan: sequence.timespan,
            metrics: this.calculateSequenceMetrics(enhancedImages)
        };
    }

calculateImageSimilarity(img1, img2) {
        // Compare aspect ratios
        const arDiff = Math.abs(img1.visualFeatures.aspectRatio - img2.visualFeatures.aspectRatio);
        const arSimilarity = 1 - Math.min(arDiff / 0.5, 1);

        // Compare visual features and color profiles combined
        const colorSimilarity = img1.imageStats.means.reduce((sum, mean, i) => {
            const diff = Math.abs(mean - img2.imageStats.means[i]) / 255;
            return sum + (1 - diff);
        }, 0) / img1.imageStats.means.length;

        // Compare vision API labels
        const labelSimilarity = this.compareLabelSets(
            img1.visionAnalysis.labels,
            img2.visionAnalysis.labels
        );

        // Weighted combination
        return (colorSimilarity * 0.5) + (arSimilarity * 0.2) + (labelSimilarity * 0.3);
    }

    compareLabelSets(labels1, labels2) {
        const set1 = new Set(labels1.map(l => l.description.toLowerCase()));
        const set2 = new Set(labels2.map(l => l.description.toLowerCase()));
        
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        
        return intersection.size / union.size;
    }

    calculateProgress(firstImage, currentImage) {
        // Calculate ink density change
        const initialDensity = this.calculateInkDensity(firstImage.imageStats);
        const currentDensity = this.calculateInkDensity(currentImage.imageStats);
        
        return {
            densityReduction: Math.max(0, (initialDensity - currentDensity) / initialDensity),
            colorFading: this.calculateColorFading(firstImage.imageStats, currentImage.imageStats),
            timeElapsed: new Date(currentImage.metadata.modified) - new Date(firstImage.metadata.modified)
        };
    }

    calculateInkDensity(imageStats) {
        return 1 - (imageStats.means.reduce((a, b) => a + b, 0) / (imageStats.means.length * 255));
    }

    calculateColorFading(stats1, stats2) {
        const intensity1 = stats1.means.reduce((a, b) => a + b, 0) / stats1.means.length;
        const intensity2 = stats2.means.reduce((a, b) => a + b, 0) / stats2.means.length;
        
        return Math.max(0, (intensity2 - intensity1) / 255);
    }

    analyzeSequence(images) {
        const progressions = [];
        let overallProgress = 0;

        for (let i = 1; i < images.length; i++) {
            const current = images[i];
            const previous = images[i - 1];
            
            const progression = {
                timeframe: {
                    start: previous.metadata.modified,
                    end: current.metadata.modified,
                    durationDays: Math.floor(
                        (new Date(current.metadata.modified) - new Date(previous.metadata.modified)) 
                        / (1000 * 60 * 60 * 24)
                    )
                },
                changes: this.calculateChanges(previous, current)
            };

            progressions.push(progression);
            overallProgress += progression.changes.totalChange;
        }

        return {
            progressions,
            overallProgress: overallProgress / (images.length - 1),
            timeline: this.analyzeTimeline(images),
            patterns: this.detectPatterns(progressions)
        };
    }

    calculateChanges(img1, img2) {
        const densityChange = this.calculateInkDensity(img1.imageStats) - 
                            this.calculateInkDensity(img2.imageStats);
        const colorChange = this.calculateColorFading(img1.imageStats, img2.imageStats);
        
        return {
            density: densityChange,
            color: colorChange,
            totalChange: (densityChange + colorChange) / 2
        };
    }

    analyzeTimeline(images) {
        const intervals = [];
        for (let i = 1; i < images.length; i++) {
            const interval = new Date(images[i].metadata.modified) - 
                           new Date(images[i-1].metadata.modified);
            intervals.push(interval);
        }

        return {
            averageInterval: intervals.reduce((a, b) => a + b, 0) / intervals.length,
            minInterval: Math.min(...intervals),
            maxInterval: Math.max(...intervals),
            consistency: this.calculateIntervalConsistency(intervals)
        };
    }

    calculateIntervalConsistency(intervals) {
        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const variance = intervals.reduce((sum, interval) => 
            sum + Math.pow(interval - mean, 2), 0) / intervals.length;
        return 1 - Math.min(Math.sqrt(variance) / mean, 1);
    }

    detectPatterns(progressions) {
        return {
            rateOfChange: this.analyzeRateOfChange(progressions),
            consistencyScore: this.calculateConsistencyScore(progressions),
            trends: this.identifyTrends(progressions)
        };
    }

    analyzeRateOfChange(progressions) {
        const rates = progressions.map(p => p.changes.totalChange / p.timeframe.durationDays);
        const averageRate = rates.reduce((a, b) => a + b, 0) / rates.length;
        
        return {
            average: averageRate,
            acceleration: this.calculateAcceleration(rates),
            consistency: this.calculateRateConsistency(rates)
        };
    }

    calculateAcceleration(rates) {
        const changes = [];
        for (let i = 1; i < rates.length; i++) {
            changes.push(rates[i] - rates[i-1]);
        }
        return changes.reduce((a, b) => a + b, 0) / changes.length;
    }

    calculateRateConsistency(rates) {
        const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
        const variance = rates.reduce((sum, rate) => 
            sum + Math.pow(rate - mean, 2), 0) / rates.length;
        return 1 - Math.min(Math.sqrt(variance) / mean, 1);
    }

    calculateConsistencyScore(progressions) {
        const timeConsistency = this.analyzeTimeline(
            progressions.map(p => ({ metadata: { modified: p.timeframe.end } }))
        ).consistency;
        
        const changeConsistency = this.calculateRateConsistency(
            progressions.map(p => p.changes.totalChange)
        );

        return (timeConsistency + changeConsistency) / 2;
    }

    identifyTrends(progressions) {
        const changes = progressions.map(p => p.changes.totalChange);
        const intervals = progressions.map(p => p.timeframe.durationDays);
        
        return {
            overallTrend: this.calculateTrend(changes),
            timeIntervalTrend: this.calculateTrend(intervals),
            effectiveness: this.calculateEffectiveness(changes, intervals)
        };
    }

    calculateTrend(values) {
        const n = values.length;
        if (n < 2) return 0;

        const xMean = (n - 1) / 2;
        const yMean = values.reduce((a, b) => a + b, 0) / n;

        let numerator = 0;
        let denominator = 0;

        for (let i = 0; i < n; i++) {
            const x = i - xMean;
            const y = values[i] - yMean;
            numerator += x * y;
            denominator += x * x;
        }

        return denominator === 0 ? 0 : numerator / denominator;
    }

    calculateEffectiveness(changes, intervals) {
        const totalChange = changes.reduce((a, b) => a + b, 0);
        const totalTime = intervals.reduce((a, b) => a + b, 0);
        
        return {
            changePerDay: totalChange / totalTime,
            consistency: this.calculateConsistencyMetric(changes, intervals)
        };
    }

    calculateConsistencyMetric(changes, intervals) {
        const rateChanges = changes.map((change, i) => change / intervals[i]);
        const mean = rateChanges.reduce((a, b) => a + b, 0) / rateChanges.length;
        const variance = rateChanges.reduce((sum, rate) => 
            sum + Math.pow(rate - mean, 2), 0) / rateChanges.length;
        
        return 1 - Math.min(Math.sqrt(variance) / mean, 1);
    }
}

// Create and export the enhanced components
const imageCacheManager = new ImageCacheManager();
const sequenceDetector = new SequenceDetector();


function validateEnvVariables() {
  const required = [
    'OPENAI_API_KEY',
    'OPENAI_ASSISTANT_ID',
    'ORGANIZATION_ID',
    'SUPABASE_URL',
    'SUPABASE_KEY'
  ];

  const missing = required.filter(key => !process.env[key]);
  if (missing.length) {
    console.error('Missing required environment variables:', missing);
    return false;
  }
  return true;
}

const Redis = require('redis');
const { promisify } = require('util');

// Initialize Redis client
const redis = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Promisify Redis methods
const redisGet = promisify(redis.get).bind(redis);
const redisSet = promisify(redis.set).bind(redis);

// Cache duration - 30 days
const CACHE_DURATION = 60 * 60 * 24 * 30;

// Initialize clients
const visionClient = new vision.ImageAnnotatorClient({
  keyFilename: path.join(__dirname, process.env.GOOGLE_APPLICATION_CREDENTIALS),
});

// Add this near the top of server.js to check configuration
console.log('Supabase Configuration:', {
  hasUrl: !!process.env.SUPABASE_URL,
  hasKey: !!process.env.SUPABASE_KEY,
  url: process.env.SUPABASE_URL,
  // Don't log the full key for security
  keyPreview: process.env.SUPABASE_KEY ? `${process.env.SUPABASE_KEY.slice(0, 8)}...` : 'missing'
});

// Modify the Supabase client initialization
const supabase = createClient(
  process.env.SUPABASE_URL || 'missing-url',
  process.env.SUPABASE_KEY || 'missing-key',
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false
    },
    global: {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
    }
  }
);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  organization: process.env.ORGANIZATION_ID,
  maxRetries: 3,
  timeout: 30000
});

// Thread management
const activeThreads = new Map();

// Directory setup
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Apply rate limiting middleware
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 500, // Limit each IP to 500 requests per windowMs
  message: { error: 'Too many requests, please try again later.' },
});

// Middleware
app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', { error: err.message, stack: err.stack });
  res.status(500).json({ error: 'Internal server error' });
});
app.use(limiter);


// Process Uncaught Exceptions and Rejections
process.on('uncaughtException', (error) => logger.error('Uncaught Exception:', error));
process.on('unhandledRejection', (reason) => logger.error('Unhandled Rejection:', reason));


// Multer config (keeping our working version)
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    console.log('Setting destination');
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    console.log('Setting filename');
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only images are allowed'));
    }
  },
});

const quickFeatures = {
  async extract(imagePath) {
    const [result] = await visionClient.labelDetection(imagePath);
    return {
      labels: result.labelAnnotations.map(l => l.description.toLowerCase()),
      mainColors: await this.extractMainColors(imagePath),
      tattooScore: this.calculateTattooScore(result.labelAnnotations)
    };
  },

  async extractMainColors(imagePath) {
    const image = await sharp(imagePath);
    const { dominant } = await image.stats();
    return dominant;
  },

  calculateTattooScore(labels) {
    const tattooKeywords = ['tattoo', 'ink', 'skin', 'art', 'body'];
    return labels.reduce((score, label) => {
      return score + (tattooKeywords.some(keyword => 
        label.description.toLowerCase().includes(keyword)) ? 1 : 0);
    }, 0);
  },

  async quickCompare(features1, features2) {
    // Quick comparison based on labels and colors
    const labelMatch = features1.labels.some(l1 => 
      features2.labels.some(l2 => l2.includes(l1) || l1.includes(l2)));
    
    const colorSimilarity = this.compareColors(features1.mainColors, features2.mainColors);
    
    return (labelMatch ? 0.6 : 0) + (colorSimilarity * 0.4);
  },

  compareColors(color1, color2) {
    const diff = Math.abs(color1.r - color2.r) + 
                Math.abs(color1.g - color2.g) + 
                Math.abs(color1.b - color2.b);
    return 1 - (diff / (3 * 255));
  }
};

// Ensure Uploads Directory Exists
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads', { recursive: true });
}

// Helper functions
async function getOrCreateThread(userId) {
  if (activeThreads.has(userId)) {
    return activeThreads.get(userId);
  }
  const thread = await openai.beta.threads.create();
  activeThreads.set(userId, thread.id);
  return thread.id;
}

function generateUUID(input) {
  return crypto.createHash('sha256').update(input).digest('hex').slice(0, 32);
}

async function checkSupabaseConnection() {
  try {
    const { data, error } = await supabase.from('processed_images').select('count');
    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Supabase connection error:', error);
    return false;
  }
}

function getCacheWithExpiry(key) {
  const item = imageSignatureCache.get(key);
  if (!item) return null;

  if (Date.now() > item.expiry) {
    imageSignatureCache.delete(key);
    return null;
  }
  return item.value;
}

// Add this helper function
function setCacheWithExpiry(key, value) {
  imageSignatureCache.set(key, {
    value,
    expiry: Date.now() + cacheTTL
  });
}


const withTimeout = async (promise, ms) => {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`Operation timed out after ${ms}ms`));
    }, ms);
  });

  try {
    const result = await Promise.race([promise, timeoutPromise]);
    clearTimeout(timeoutId);
    return result;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
};

const processBatchWithLimit = async (items, batchSize, processor, concurrencyLimit = 10) => {
  const results = [];
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, Math.min(i + batchSize, items.length));
    const batchResults = await Promise.all(
      batch.slice(0, concurrencyLimit).map(processor)
    );
    results.push(...batchResults.filter(Boolean));
    if (results.some(r => r.similarity > 0.95)) break;
  }
  return results;
};

async function quickCompare(sig1, sig2) {
  // Compare aspect ratios
  const aspectRatioDiff = Math.abs(sig1.aspectRatio - sig2.aspectRatio);
  const aspectRatioScore = 1 - Math.min(aspectRatioDiff, 1);

  // Compare color averages
  const colorScore = sig1.channelAverages.reduce((sum, avg, i) => {
    const diff = Math.abs(avg - sig2.channelAverages[i]) / 255;
    return sum + (1 - diff);
  }, 0) / sig1.channelAverages.length;

  return (aspectRatioScore * 0.3) + (colorScore * 0.7);
}

async function getOrCreateAssistant() {
  try {
    // Try to retrieve existing assistant
    if (process.env.OPENAI_ASSISTANT_ID) {
      try {
        const assistant = await openai.beta.assistants.retrieve(
          process.env.OPENAI_ASSISTANT_ID
        );
        console.log('Retrieved existing assistant:', assistant.id);
        return assistant;
      } catch (error) {
        console.log('Could not retrieve assistant:', error.message);
      }
    }

    // Create new assistant if needed
    const assistant = await openai.beta.assistants.create({
      name: "Tatt2AwAI",
      instructions: "You are Tatt2AwAI, a specialized assistant focused on tattoo removal and medical information. You have access to search and analyze images in the connected Dropbox account.",
      tools: [{ type: "code_interpreter" }],
      model: "gpt-4-turbo"
    });

    console.log('Created new assistant:', assistant.id);
    return assistant;
  } catch (error) {
    console.error('Error with assistant:', error);
    throw error;
  }
}

async function findRelatedSequences(imagePath) {
  if (!imagePath) return [];
  
  try {
    const directory = path.dirname(imagePath);
    const entries = await dropboxManager.fetchDropboxEntries(directory);
    
    const imageFiles = entries.result.entries
      .filter(entry => ['.jpg', '.jpeg', '.png'].some(ext => 
        entry.path_lower.endsWith(ext)
      ))
      .sort((a, b) => new Date(a.server_modified) - new Date(b.server_modified));

    if (imageFiles.length < 2) return [];

    return [{
      directory,
      images: imageFiles.map(file => ({
        path: file.path_lower,
        timestamp: file.server_modified,
        metadata: {
          size: file.size,
          name: file.name
        }
      })),
      count: imageFiles.length,
      timespan: {
        start: imageFiles[0].server_modified,
        end: imageFiles[imageFiles.length - 1].server_modified
      }
    }];
  } catch (error) {
    console.error('Error finding sequences:', error);
    return [];
  }
}

async function processImageWithAnalysis(file) {
  console.log('Processing image:', file.filename);
  
  const labels = await visionClient.labelDetection(file.path);
  const processedImage = await imageProcessor.processImage({ path: file.path });
  
  // Add to knowledge base
  await knowledgeBase.addDocument(
    crypto.randomUUID(),
    JSON.stringify({
      analysis: labels[0],
      processed: processedImage
    }),
    {
      type: 'image',
      path: file.path,
      timestamp: new Date().toISOString()
    }
  );

  return {
    labels: labels[0].labelAnnotations,
    processed: processedImage
  };
}

const tattooAnalyzer = {
    analyzeImage: async function(imageData, originalPath) {
try {
        const validation = this.validateAndClassifyImage(imageData);
        
        if (!validation.isTattoo) {
            return {
                path: originalPath,
                timestamp: new Date().toISOString(),
                isTattoo: false,
                confidence: validation.confidence,
                reason: validation.reason || 'Not a tattoo image',
                detectedFeatures: validation.characteristics || {},
                generalImageInfo: {
                    format: imageData.metadata?.format || 'unknown',
                    dimensions: {
                        width: imageData.metadata?.width || 0,
                        height: imageData.metadata?.height || 0
                    }
                }
            };
        }


        // First check if this is likely a tattoo image
const isTattoo = this.isLikelyTattoo(imageData.visionAnalysis);
        
        let stages = null;
        if (isTattoo) {
            try {
                stages = {
                    before: this.detectOriginalTattoo(imageData),
                    during: this.detectTreatmentStage(imageData),
                    after: this.detectRemovalOutcome(imageData)
                };
            } catch (error) {
                console.log('Error detecting stages:', error);
                stages = {
                    before: null,
                    during: null,
                    after: null
                };
            }
        }

 const analysis = {
            path: originalPath,
            timestamp: new Date().toISOString(),
            isTattoo: isTattoo,
            stages: stages,
            metrics: isTattoo ? await this.calculateMetrics(imageData) : null,
            patterns: isTattoo ? this.detectPatterns(imageData) : null,
            sequence: {
                isPartOfSequence: false,
                sequencePosition: null,
                relativePath: null
            },
            generalImageInfo: {
                format: imageData.metadata?.format || 'unknown',
                dimensions: {
                    width: imageData.metadata?.width || 0,
                    height: imageData.metadata?.height || 0
                },
                dominantColors: imageData.stats?.dominant ? [imageData.stats.dominant] : [],
                labels: imageData.visionAnalysis?.labelAnnotations || []
            }
        };

        return analysis;
    } catch (error) {
        console.error('Error in analyzeImage:', error);
        return {
            path: originalPath,
            timestamp: new Date().toISOString(),
            error: error.message,
            isTattoo: false,
            generalImageInfo: {
                format: imageData.metadata?.format || 'unknown',
                dimensions: {
                    width: imageData.metadata?.width || 0,
                    height: imageData.metadata?.height || 0
                }
            }
        };
    }
},

isLikelyTattoo: function(visionAnalysis) {
    if (!visionAnalysis?.labelAnnotations) return false;
    
    const tattooKeywords = ['tattoo', 'ink', 'skin art', 'body art', 'tribal'];
    return visionAnalysis.labelAnnotations.some(label => 
        tattooKeywords.some(keyword => 
            label.description.toLowerCase().includes(keyword)
        )
    );
},

// Add these methods to your existing tattooAnalyzer object
// Add to your tattooAnalyzer object

validateAndClassifyImage: function(imageData) {
    try {
        if (!imageData || !imageData.visionAnalysis) {
            return {
                isTattoo: false,
                confidence: 0,
                reason: 'Missing image data or vision analysis'
            };
        }

        const labels = imageData.visionAnalysis.labelAnnotations || [];
        const tattooKeywords = ['tattoo', 'ink', 'body art', 'skin art', 'tribal'];
        
        // Check for tattoo-specific labels
        const tattooLabels = labels.filter(label => 
            tattooKeywords.some(keyword => 
                label.description.toLowerCase().includes(keyword)
            )
        );

        // If no tattoo labels found, it's likely not a tattoo
        if (tattooLabels.length === 0) {
            return {
                isTattoo: false,
                confidence: 0,
                reason: 'No tattoo-related features detected',
                detectedLabels: labels.map(l => l.description)
            };
        }

        // Verify additional characteristics
        const hasInkCharacteristics = this.verifyInkCharacteristics(imageData);
        const hasSkinContext = this.verifySkinContext(imageData);

        // Calculate overall confidence
        const confidence = this.calculateDetectionConfidence(
            tattooLabels,
            hasInkCharacteristics,
            hasSkinContext
        );

        return {
            isTattoo: confidence > 0.6,
            confidence,
            characteristics: {
                hasInkFeatures: hasInkCharacteristics,
                hasSkinContext: hasSkinContext,
                tattooLabels: tattooLabels.map(l => l.description)
            }
        };
    } catch (error) {
        console.log('Error in validateAndClassifyImage:', error);
        return {
            isTattoo: false,
            confidence: 0,
            error: error.message
        };
    }
},

verifyInkCharacteristics: function(imageData) {
    const stats = this.extractImageStats(imageData);
    if (!stats.isValid) return false;

    // Check for ink-like characteristics
    const hasHighContrast = this.checkContrast(stats);
    const hasEdgeDefinition = this.calculateEdgeDefinition(stats) > 0.4;
    const hasInkLikeColors = this.checkInkColors(stats);

    return (hasHighContrast && hasEdgeDefinition) || hasInkLikeColors;
},

verifySkinContext: function(imageData) {
    if (!imageData.visionAnalysis || !imageData.visionAnalysis.labelAnnotations) {
        return false;
    }

    const skinKeywords = ['skin', 'flesh', 'body', 'dermis'];
    const labels = imageData.visionAnalysis.labelAnnotations;

    return labels.some(label => 
        skinKeywords.some(keyword => 
            label.description.toLowerCase().includes(keyword)
        )
    );
},

calculateDetectionConfidence: function(tattooLabels, hasInkCharacteristics, hasSkinContext) {
    const labelConfidence = Math.max(...tattooLabels.map(l => l.score || 0));
    const weights = {
        labels: 0.5,
        inkCharacteristics: 0.3,
        skinContext: 0.2
    };

    return (labelConfidence * weights.labels) +
           (hasInkCharacteristics ? weights.inkCharacteristics : 0) +
           (hasSkinContext ? weights.skinContext : 0);
},

checkContrast: function(stats) {
    if (!stats.channels) return false;
    
    const contrasts = stats.channels.map(channel => {
        if (!channel.min || !channel.max) return 0;
        return (channel.max - channel.min) / 255;
    });

    return Math.max(...contrasts) > 0.4;
},

checkInkColors: function(stats) {
    if (!stats.means) return false;

    // Check for common ink color characteristics
    const [r, g, b] = stats.means;
    
    // Check for black/gray ink
    const isGrayScale = Math.abs(r - g) < 20 && Math.abs(g - b) < 20;
    const isDark = (r + g + b) / 3 < 128;

    // Check for colored ink
    const hasColorVariation = Math.max(r, g, b) - Math.min(r, g, b) > 50;

    return (isGrayScale && isDark) || hasColorVariation;
},

detectFadingPattern: function(imageData) {
    try {
        if (!this.validateImageData(imageData)) {
            return {
                hasFading: false,
                confidence: 0,
                pattern: {
                    type: 'unknown',
                    uniformity: 0,
                    intensity: 0
                }
            };
        }

        const means = this.getMeansFromStats(imageData.stats);
        if (!means) {
            return {
                hasFading: false,
                confidence: 0,
                pattern: {
                    type: 'unknown',
                    uniformity: 0,
                    intensity: 0
                }
            };
        }

        const inkDensity = this.calculateInkDensity(means);
        const uniformity = this.calculateFadingUniformity(imageData);
        const patternType = this.determineFadingPatternType(imageData.stats);

        return {
            hasFading: inkDensity < TATTOO_ANALYSIS_THRESHOLDS.INK_DENSITY.HIGH,
            confidence: Math.min((1 - inkDensity) * 2, 1),
            pattern: {
                type: patternType,
                uniformity: uniformity,
                intensity: 1 - inkDensity
            },
            areas: this.detectFadingAreas(imageData.stats)
        };
    } catch (error) {
        console.log('Error in detectFadingPattern:', error);
        return {
            hasFading: false,
            confidence: 0,
            pattern: {
                type: 'unknown',
                uniformity: 0,
                intensity: 0
            },
            error: error.message
        };
    }
},

validateStats: function(stats) {
    if (!stats) {
        console.log('Stats object is undefined');
        return null;
    }

    // Ensure channels exist and have required properties
    if (!stats.channels || !Array.isArray(stats.channels)) {
        console.log('Invalid channels data:', stats);
        return null;
    }

    // Create validated channels with safe defaults
    const validatedChannels = stats.channels.map(channel => ({
        mean: typeof channel.mean === 'number' ? channel.mean : 0,
        std: typeof channel.std === 'number' ? channel.std : 0,
        min: typeof channel.min === 'number' ? channel.min : 0,
        max: typeof channel.max === 'number' ? channel.max : 255
    }));

    return {
        channels: validatedChannels,
        means: validatedChannels.map(c => c.mean),
        stdDevs: validatedChannels.map(c => c.std),
        isValid: true
    };
},


calculateDensityFromChannels: function(channels) {
    const meanIntensity = channels.reduce((sum, channel) => 
        sum + channel.mean, 0) / channels.length;
    return 1 - (meanIntensity / 255);
},

extractImageStats: function(imageData) {
    try {
        const { stats } = imageData;
        if (!stats || !stats.channels) {
            return {
                means: [0, 0, 0],
                stdDevs: [0, 0, 0],
                isValid: false
            };
        }

        return {
            means: stats.channels.map(c => c.mean || 0),
            stdDevs: stats.channels.map(c => c.std || 0),
            channels: stats.channels,
            isValid: true
        };
    } catch (error) {
        console.log('Error extracting image stats:', error);
        return {
            means: [0, 0, 0],
            stdDevs: [0, 0, 0],
            isValid: false
        };
    }
},

analyzeInkDistribution: function(stats) {
    if (!stats || !stats.channels) return null;
    
    const channels = stats.channels.map(c => ({
        mean: c.mean || 0,
        std: c.std || 0,
        distribution: c.histogram || []
    }));

    return {
        density: this.calculateDensityFromChannels(channels),
        uniformity: this.calculateDistributionUniformity(channels),
        patterns: this.detectDistributionPatterns(channels)
    };
},

assessHealingIndicators: function(imageData) {
    const stats = this.extractImageStats(imageData);
    const redness = this.calculateRedness(stats);
    const texture = this.calculateTexture(stats.stdDevs);

    return {
        inflammationLevel: redness,
        textureScore: texture,
        healingStage: this.determineHealingStage(redness, texture),
        skinResponse: this.analyzeSkinResponse(imageData)
    };
},

determineHealingStage: function(redness, texture) {
    if (redness > 0.7) return 'acute';
    if (redness > 0.4) return 'intermediate';
    if (texture > 0.6) return 'remodeling';
    return 'resolved';
},


analyzeTextureRecovery: function(stats) {
    const textureScore = this.calculateTexture(stats.channels.map(c => c.std || 0));
    const uniformity = this.calculateDistributionUniformity(stats.channels);

    return {
        textureScore,
        uniformity,
        recovery: 1 - Math.max(textureScore, 1 - uniformity),
        patterns: this.detectTexturePatterns(stats)
    };
},

calculateStageConfidence: function(metrics, stage) {
    switch(stage) {
        case 'before':
            return (metrics.originalDensity * 0.4 +
                   metrics.edgeDefinition * 0.3 +
                   (metrics.colorProfile ? 0.3 : 0));

        case 'during':
            return (metrics.dotPatternMetrics.coverage * 0.3 +
                   metrics.fadingMetrics.uniformity * 0.4 +
                   (metrics.healingIndicators ? 0.3 : 0));

        case 'after':
            return (metrics.clearanceLevel * 0.5 +
                   metrics.textureAnalysis.recovery * 0.3 +
                   (1 - metrics.residualDensity) * 0.2);

        default:
            return 0;
    }
},

detectTexturePatterns: function(stats) {
    const patterns = [];
    const stdDevs = stats.channels.map(c => c.std || 0);
    const means = stats.channels.map(c => c.mean || 0);

    // Analyze for common texture patterns
    if (this.hasUniformTexture(stdDevs)) {
        patterns.push({
            type: 'uniform',
            confidence: this.calculateUniformityConfidence(stdDevs)
        });
    }

    if (this.hasSpottedPattern(means, stdDevs)) {
        patterns.push({
            type: 'spotted',
            confidence: this.calculateSpottedConfidence(means, stdDevs)
        });
    }

    if (this.hasGradientPattern(means)) {
        patterns.push({
            type: 'gradient',
            confidence: this.calculateGradientConfidence(means)
        });
    }

    return patterns;
},


detectRemovalProgress: function(imageData, stage) {
    try {
        if (!imageData || !imageData.stats) {
            return {
                stage: 'unknown',
                confidence: 0,
                metrics: null
            };
        }

        const stats = this.extractImageStats(imageData);
        const inkDensity = this.calculateInkDensity(stats.means);
        const dotPattern = this.detectDotPattern(imageData);
        const fadingPattern = this.detectFadingPattern(imageData);

        // Stage-specific analysis
        let stageMetrics = {};
        switch(stage) {
            case 'before':
                stageMetrics = {
                    originalDensity: inkDensity,
                    inkDistribution: this.analyzeInkDistribution(stats),
                    edgeDefinition: this.calculateEdgeDefinition(stats),
                    colorProfile: this.analyzeColorProfile(stats)
                };
                break;

            case 'during':
                stageMetrics = {
                    currentDensity: inkDensity,
                    dotPatternMetrics: {
                        coverage: dotPattern.coverage,
                        uniformity: dotPattern.uniformity,
                        intensity: dotPattern.confidence
                    },
                    fadingMetrics: {
                        pattern: fadingPattern.pattern,
                        uniformity: fadingPattern.uniformity,
                        intensity: 1 - inkDensity
                    },
                    healingIndicators: this.assessHealingIndicators(imageData)
                };
                break;

            case 'after':
                stageMetrics = {
                    residualDensity: inkDensity,
                    clearanceLevel: 1 - inkDensity,
                    skinResponse: this.analyzeSkinResponse(imageData),
                    textureAnalysis: this.analyzeTextureRecovery(stats)
                };
                break;
        }

        return {
            stage,
            confidence: this.calculateStageConfidence(stageMetrics, stage),
            metrics: stageMetrics,
            analysis: {
                inkDensity,
                dotPattern: dotPattern.confidence > 0.5 ? dotPattern : null,
                fadingPattern: fadingPattern.hasFading ? fadingPattern : null
            }
        };
    } catch (error) {
        console.log('Error in detectRemovalProgress:', error);
        return {
            stage: 'unknown',
            confidence: 0,
            metrics: null,
            error: error.message
        };
    }
},


// Add a safe stats extractor
extractImageStats: function(imageData) {
    try {
        if (!imageData || !imageData.stats || !imageData.stats.channels) {
            return {
                means: [0, 0, 0],
                stdDevs: [0, 0, 0],
                isValid: false
            };
        }

        const means = imageData.stats.channels.map(c => c.mean || 0);
        const stdDevs = imageData.stats.channels.map(c => c.std || 0);

        return {
            means,
            stdDevs,
            isValid: true
        };
    } catch (error) {
        console.log('Error extracting image stats:', error);
        return {
            means: [0, 0, 0],
            stdDevs: [0, 0, 0],
            isValid: false
        };
    }
},

determineFadingPatternType: function(stats) {
    const stdDevs = stats.channels.map(c => c.std || 0);
    const avgStdDev = stdDevs.reduce((a, b) => a + b, 0) / stdDevs.length;
    
    if (avgStdDev < 30) {
        return 'uniform';
    } else if (avgStdDev > 60) {
        return 'spotted';
    } else {
        return 'gradient';
    }
},

detectFadingAreas: function(stats) {
    // Simplified area detection based on intensity variations
    const means = stats.channels.map(c => c.mean || 0);
    const avgIntensity = means.reduce((a, b) => a + b, 0) / means.length;
    
    return {
        light: avgIntensity > 192,
        medium: avgIntensity > 128 && avgIntensity <= 192,
        dark: avgIntensity <= 128
    };
},

analyzePatternDistribution: function(imageData) {
    try {
        const { stats } = imageData;
        
        if (!stats || !stats.channels) {
            return {
                type: 'unknown',
                confidence: 0
            };
        }

        const distributionMetrics = {
            uniformity: this.calculateDistributionUniformity(stats),
            coverage: this.calculatePatternCoverage(stats),
            intensity: this.calculateAverageIntensity(stats)
        };

        return {
            type: this.determineDistributionType(distributionMetrics),
            metrics: distributionMetrics,
            confidence: this.calculateDistributionConfidence(distributionMetrics)
        };
    } catch (error) {
        console.log('Error in analyzePatternDistribution:', error);
        return {
            type: 'unknown',
            confidence: 0,
            error: error.message
        };
    }
},

calculateDistributionUniformity: function(stats) {
    const stdDevs = stats.channels.map(c => c.std || 0);
    const avgStdDev = stdDevs.reduce((a, b) => a + b, 0) / stdDevs.length;
    return Math.max(0, 1 - (avgStdDev / 128));
},

calculatePatternCoverage: function(stats) {
    const means = stats.channels.map(c => c.mean || 0);
    const avgIntensity = means.reduce((a, b) => a + b, 0) / means.length;
    return 1 - (avgIntensity / 255);
},

calculateAverageIntensity: function(stats) {
    const means = stats.channels.map(c => c.mean || 0);
    return means.reduce((a, b) => a + b, 0) / (means.length * 255);
},

determineDistributionType: function(metrics) {
    if (metrics.uniformity > 0.8) {
        return 'even';
    } else if (metrics.coverage > 0.7) {
        return 'dense';
    } else if (metrics.intensity < 0.3) {
        return 'sparse';
    } else {
        return 'mixed';
    }
},

calculateDistributionConfidence: function(metrics) {
    // Combine multiple metrics for overall confidence
    const weights = {
        uniformity: 0.4,
        coverage: 0.3,
        intensity: 0.3
    };

    return (
        metrics.uniformity * weights.uniformity +
        metrics.coverage * weights.coverage +
        (1 - metrics.intensity) * weights.intensity
    );
},

detectOriginalTattoo: function(imageData) {
    if (!imageData || !imageData.stats) {
        console.log('Invalid image data for original tattoo detection:', imageData);
        return {
            isOriginal: false,
            confidence: 0,
            characteristics: {
                density: 0,
                clarity: 0,
                isBlackInk: false
            }
        };
    }

    const means = this.getMeansFromStats(imageData.stats);
    const inkDensity = this.calculateInkDensity(means);
    const edgeClarity = this.calculateEdgeClarity(imageData.stats.channels.map(c => c.std || 0));

    return {
        isOriginal: inkDensity > TATTOO_ANALYSIS_THRESHOLDS.INK_DENSITY.HIGH,
        confidence: inkDensity * edgeClarity,
        characteristics: {
            density: inkDensity,
            clarity: edgeClarity,
            isBlackInk: this.isBlackInk(means)
        }
    };
},

calculateEdgeDefinition: function(stats) {
    try {
        if (!stats || !stats.channels) return 0;
        
        // Calculate edge definition using sharpness and standard deviation
        const avgStdDev = stats.channels.reduce((sum, channel) => 
            sum + (channel.std || 0), 0) / stats.channels.length;
        
        // Higher std dev indicates more defined edges
        return Math.min(1, avgStdDev / 128);
    } catch (error) {
        console.log('Error calculating edge definition:', error);
        return 0;
    }
},

// Add these to the tattooAnalyzer object
calculateRedness: function(stats) {
    try {
        if (!stats || !stats.channels || stats.channels.length < 3) return 0;
        
        // Compare red channel to others to detect redness
        const [red, green, blue] = stats.channels;
        const redDominance = (red.mean - ((green.mean + blue.mean) / 2)) / 255;
        return Math.max(0, Math.min(1, redDominance));
    } catch (error) {
        console.log('Error calculating redness:', error);
        return 0;
    }
},

calculateTexture: function(stdDevs) {
    try {
        if (!Array.isArray(stdDevs)) return 0;
        const avgStdDev = stdDevs.reduce((sum, std) => sum + std, 0) / stdDevs.length;
        return Math.min(1, avgStdDev / 128);
    } catch (error) {
        console.log('Error calculating texture:', error);
        return 0;
    }
},

assessHealing: function(imageData) {
    try {
        const redness = this.calculateRedness(imageData.stats);
        const texture = this.calculateTexture(imageData.stats.channels.map(c => c.std));
        
        return {
            stage: redness > 0.5 ? 'active' : 'healed',
            rednessLevel: redness,
            textureScore: texture,
            overallScore: (1 - redness) * (1 - texture)
        };
    } catch (error) {
        console.log('Error assessing healing:', error);
        return {
            stage: 'unknown',
            rednessLevel: 0,
            textureScore: 0,
            overallScore: 0
        };
    }
},

calculatePatternDensity: function(visionAnalysis) {
    try {
        if (!visionAnalysis || !visionAnalysis.localizedObjectAnnotations) {
            return 0;
        }

        // Calculate density based on number and size of detected objects
        const objects = visionAnalysis.localizedObjectAnnotations;
        if (!objects.length) return 0;

        const totalArea = objects.reduce((sum, obj) => {
            if (!obj.boundingPoly?.normalizedVertices) return sum;
            const vertices = obj.boundingPoly.normalizedVertices;
            const area = this.calculatePolygonArea(vertices);
            return sum + area;
        }, 0);

        return Math.min(1, totalArea * objects.length / 10);
    } catch (error) {
        console.log('Error calculating pattern density:', error);
        return 0;
    }
},

calculatePolygonArea: function(vertices) {
    if (!vertices || vertices.length < 3) return 0;
    
    let area = 0;
    for (let i = 0; i < vertices.length; i++) {
        const j = (i + 1) % vertices.length;
        area += vertices[i].x * vertices[j].y;
        area -= vertices[j].x * vertices[i].y;
    }
    return Math.abs(area) / 2;
},

analyzeColorProfile: function(stats) {
    try {
        if (!stats || !stats.channels) return null;
        
        return {
            channels: stats.channels.map((channel, index) => ({
                color: ['red', 'green', 'blue'][index],
                mean: channel.mean,
                std: channel.std,
                distribution: channel.histogram
            })),
            dominantColor: stats.dominant,
            uniformity: this.calculateColorUniformity(stats.channels)
        };
    } catch (error) {
        console.log('Error analyzing color profile:', error);
        return null;
    }
},

calculateColorUniformity: function(channels) {
    try {
        if (!channels || !channels.length) return 0;
        const stdDevs = channels.map(c => c.std || 0);
        const avgStdDev = stdDevs.reduce((a, b) => a + b, 0) / stdDevs.length;
        return 1 - Math.min(1, avgStdDev / 128);
    } catch (error) {
        console.log('Error calculating color uniformity:', error);
        return 0;
    }
},

calculateDimensions: function(imageData) {
    try {
        if (!imageData.metadata) return null;
        
        return {
            width: imageData.metadata.width,
            height: imageData.metadata.height,
            aspectRatio: imageData.metadata.width / imageData.metadata.height,
            resolution: imageData.metadata.density
        };
    } catch (error) {
        console.log('Error calculating dimensions:', error);
        return null;
    }
},

    detectTreatmentStage: function(imageData) {
        const dotPattern = this.detectDotPattern(imageData);
        const fadingLevel = this.calculateFadingLevel(imageData);

        return {
            isDuringTreatment: dotPattern.confidence > TATTOO_ANALYSIS_THRESHOLDS.PATTERN_DETECTION.DOT_CONFIDENCE,
            pattern: {
                dotCount: dotPattern.count,
                coverage: dotPattern.coverage,
                uniformity: dotPattern.uniformity
            },
            fading: {
                level: fadingLevel,
                uniformity: this.calculateFadingUniformity(imageData),
                areas: this.detectFadingAreas(imageData)
            }
        };
    },

calculateFadingUniformity: function(imageData) {
    try {
        if (!imageData || !imageData.stats || !imageData.stats.channels) {
            return 0;
        }
        // Calculate uniformity based on standard deviation of channels
        const stdDevs = imageData.stats.channels.map(c => c.std || 0);
        const avgStdDev = stdDevs.reduce((a, b) => a + b, 0) / stdDevs.length;
        // Higher std dev means less uniform
        return Math.max(0, 1 - (avgStdDev / 128));
    } catch (error) {
        console.log('Error calculating fading uniformity:', error);
        return 0;
    }
},

detectFadingAreas: function(imageData) {
    try {
        if (!imageData || !imageData.stats) {
            return [];
        }
        return {
            hasFading: true,
            areas: []  // Placeholder for actual fading area detection
        };
    } catch (error) {
        console.log('Error detecting fading areas:', error);
        return [];
    }
},

calculateFadingPercentage: function(imageData) {
    try {
        if (!imageData || !imageData.stats || !imageData.stats.channels) {
            return 0;
        }
        const avgIntensity = imageData.stats.channels.reduce((sum, channel) => 
            sum + channel.mean, 0) / imageData.stats.channels.length;
        // Higher intensity means more fading
        return Math.min(100, (avgIntensity / 255) * 100);
    } catch (error) {
        console.log('Error calculating fading percentage:', error);
        return 0;
    }
},

    detectRemovalOutcome: function(imageData) {
        const inkDensity = this.calculateInkDensity(imageData.stats.means);
        const skinResponse = this.analyzeSkinResponse(imageData);

        return {
            isComplete: inkDensity < TATTOO_ANALYSIS_THRESHOLDS.INK_DENSITY.MINIMAL,
            fadingPercentage: this.calculateFadingPercentage(imageData),
            skinCondition: skinResponse
        };
    },

    calculateMetrics: async function(imageData) {
        const { stats, visionAnalysis } = imageData;
        
        return {
            inkDensity: this.calculateInkDensity(stats.means),
            patternDensity: this.calculatePatternDensity(visionAnalysis),
            edgeDefinition: this.calculateEdgeDefinition(stats),
            colorProfile: this.analyzeColorProfile(stats),
            dimensions: this.calculateDimensions(imageData)
        };
    },

    detectPatterns: function(imageData) {
        const dotPattern = this.detectDotPattern(imageData);
        const fadingPattern = this.detectFadingPattern(imageData);

        return {
            hasDotPattern: dotPattern.confidence > TATTOO_ANALYSIS_THRESHOLDS.PATTERN_DETECTION.DOT_CONFIDENCE,
            dotPattern,
            fadingPattern,
            distribution: this.analyzePatternDistribution(imageData)
        };
    },
validateImageData: function(imageData) {
    if (!imageData) {
        console.log('Missing image data');
        return false;
    }

    if (!imageData.stats || !imageData.stats.channels) {
        console.log('Missing image stats or channels');
        return false;
    }

    const hasValidChannels = imageData.stats.channels.every(channel => 
        typeof channel.mean === 'number' && 
        typeof channel.std === 'number'
    );

    if (!hasValidChannels) {
        console.log('Invalid channel data detected');
        return false;
    }

    return true;
},

getMeansFromStats: function(stats) {
    if (!stats || !stats.channels || !Array.isArray(stats.channels)) {
        console.log('Invalid stats data:', stats);
        return null;
    }

    const means = stats.channels.map(channel => {
        if (typeof channel.mean !== 'number') {
            console.log('Invalid channel mean:', channel);
            return null;
        }
        return channel.mean;
    });

    if (means.some(mean => mean === null)) {
        console.log('Some channel means are invalid');
        return null;
    }

    return means;
},

    // Core analysis functions
calculateInkDensity: function(means) {
    try {
        // Handle array input
        if (Array.isArray(means)) {
            // Validate array elements
            const validMeans = means.filter(m => typeof m === 'number' && !isNaN(m));
            if (validMeans.length === 0) {
                console.log('No valid means values in array');
                return 0;
            }
            const average = validMeans.reduce((sum, val) => sum + val, 0) / validMeans.length;
            return Math.max(0, Math.min(1, 1 - (average / 255)));
        }

        // Handle stats object input
        if (means && typeof means === 'object') {
            const stats = this.validateStats(means);
            if (!stats || !stats.means.length) {
                console.log('Invalid stats object for ink density calculation');
                return 0;
            }
            const average = stats.means.reduce((sum, val) => sum + val, 0) / stats.means.length;
            return Math.max(0, Math.min(1, 1 - (average / 255)));
        }

        console.log('Invalid input type for calculateInkDensity:', typeof means);
        return 0;
    } catch (error) {
        console.log('Error calculating ink density:', error);
        return 0;
    }
},

// Add a safe image stats extractor
extractImageStats: function(imageData) {
    try {
        if (!imageData) {
            console.log('No image data provided');
            return this.createDefaultStats();
        }

        const stats = imageData.stats;
        if (!stats) {
            console.log('No stats in image data');
            return this.createDefaultStats();
        }

        const validatedStats = this.validateStats(stats);
        if (!validatedStats) {
            console.log('Invalid stats data');
            return this.createDefaultStats();
        }

        return validatedStats;
    } catch (error) {
        console.log('Error extracting image stats:', error);
        return this.createDefaultStats();
    }
},

createDefaultStats: function() {
    return {
        channels: [
            { mean: 0, std: 0, min: 0, max: 255 },
            { mean: 0, std: 0, min: 0, max: 255 },
            { mean: 0, std: 0, min: 0, max: 255 }
        ],
        means: [0, 0, 0],
        stdDevs: [0, 0, 0],
        isValid: false
    };
},

// And add this helper function to get means from stats
getMeansFromStats: function(stats) {
    if (!stats || !stats.channels) return [];
    return stats.channels.map(channel => channel.mean || 0);
},
    calculateEdgeClarity: function(stdDevs) {
        // Higher std dev indicates clearer edges
        return Math.min(...stdDevs.map(std => std / 128), 1);
    },

    isBlackInk: function(means) {
        const threshold = 100; // Black ink threshold
        return means.every(mean => mean < threshold);
    },

extractDotFeatures: function(visionAnalysis) {
    try {
        const dots = [];
        // Look for dot patterns in the image
        if (visionAnalysis.localizedObjectAnnotations) {
            visionAnalysis.localizedObjectAnnotations.forEach(obj => {
                if (obj.boundingPoly && obj.boundingPoly.normalizedVertices) {
                    dots.push({
                        position: obj.boundingPoly.normalizedVertices,
                        confidence: obj.score
                    });
                }
            });
        }
        return dots;
    } catch (error) {
        console.log('Error extracting dot features:', error);
        return [];
    }
},

detectDotPattern: function(imageData) {
    try {
        const { stats, visionAnalysis } = imageData;
        const dotFeatures = this.extractDotFeatures(visionAnalysis);
        
        return {
            confidence: this.calculateDotConfidence(dotFeatures),
            count: dotFeatures.length,
            coverage: this.calculatePatternCoverage(dotFeatures),
            uniformity: this.calculatePatternUniformity(dotFeatures)
        };
    } catch (error) {
        console.log('Error detecting dot pattern:', error);
        return {
            confidence: 0,
            count: 0,
            coverage: 0,
            uniformity: 0
        };
    }
},

// Add these additional helper functions
calculateDotConfidence: function(dotFeatures) {
    if (!dotFeatures || !dotFeatures.length) return 0;
    return dotFeatures.reduce((sum, dot) => sum + (dot.confidence || 0), 0) / dotFeatures.length;
},

calculatePatternCoverage: function(dotFeatures) {
    if (!dotFeatures || !dotFeatures.length) return 0;
    // Simple coverage calculation based on number of dots
    return Math.min(dotFeatures.length / 100, 1);
},

calculatePatternUniformity: function(dotFeatures) {
    if (!dotFeatures || dotFeatures.length < 2) return 0;
    try {
        // Calculate average distance between dots
        let totalDistance = 0;
        let count = 0;
        
        for (let i = 0; i < dotFeatures.length; i++) {
            for (let j = i + 1; j < dotFeatures.length; j++) {
                const dist = this.calculateDistance(
                    dotFeatures[i].position[0],
                    dotFeatures[j].position[0]
                );
                totalDistance += dist;
                count++;
            }
        }
        
        return count > 0 ? 1 - (totalDistance / count) : 0;
    } catch (error) {
        console.log('Error calculating pattern uniformity:', error);
        return 0;
    }
},

calculateDistance: function(point1, point2) {
    if (!point1 || !point2) return 1;
    return Math.sqrt(
        Math.pow((point2.x || 0) - (point1.x || 0), 2) +
        Math.pow((point2.y || 0) - (point1.y || 0), 2)
    );
},

    calculateFadingLevel: function(imageData) {
        const inkDensity = this.calculateInkDensity(imageData.stats.means);
        if (inkDensity < TATTOO_ANALYSIS_THRESHOLDS.INK_DENSITY.MINIMAL) return 'complete';
        if (inkDensity < TATTOO_ANALYSIS_THRESHOLDS.INK_DENSITY.LOW) return 'significant';
        if (inkDensity < TATTOO_ANALYSIS_THRESHOLDS.INK_DENSITY.MEDIUM) return 'moderate';
        return 'minimal';
    },

    analyzeSkinResponse: function(imageData) {
        // Analyze skin condition and healing response
        const { means, stdDevs } = imageData.stats;
        
        return {
            redness: this.calculateRedness(means),
            texture: this.calculateTexture(stdDevs),
            healing: this.assessHealing(imageData)
        };
    }
};

// Add after tattooAnalyzer
async function initializeImageCache() {
    const startTime = Date.now();
    const BATCH_SIZE = 5;
    const processedEntries = new Set();
    const failedEntries = new Set();

    try {
        initializationProgress.status = 'starting';
        initializationProgress.startTime = startTime;
        logger.info('Starting complete image cache initialization...');

        // Check for existing progress and failures in Supabase
        const { data: existingProgress } = await supabase
            .from('initialization_progress')
            .select('*')
            .single();

        // Ensure Dropbox connection
        const dropboxStatus = await dropboxManager.ensureAuth();
        if (!dropboxStatus) {
            throw new Error('Failed to authenticate with Dropbox');
        }

        // Get all Dropbox images with validation
        const entries = await dropboxManager.fetchDropboxEntries('');
        if (!entries?.result?.entries) {
            throw new Error('No entries returned from Dropbox');
        }

        const imageFiles = entries.result.entries.filter(entry => 
            ['.jpg', '.jpeg', '.png'].some(ext => entry.path_lower.endsWith(ext))
        );

        // Initialize progress tracking
        initializationProgress.total = imageFiles.length;
        logger.info(`Processing ${imageFiles.length} images for complete analysis...`);

        // Process in batches
        for (let i = 0; i < imageFiles.length; i += BATCH_SIZE) {
            const batch = imageFiles.slice(i, i + BATCH_SIZE);
            
            await Promise.all(batch.map(async (entry) => {
                const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(entry.path_lower)}`);
                
                try {
                    // Skip if already successfully processed
                    if (existingProgress?.processed_paths?.includes(entry.path_lower)) {
                        processedEntries.add(entry.path_lower);
                        return;
                    }

                    // Download and process file
                    const fileData = await dropboxManager.downloadFile(entry.path_lower);
                    if (!fileData?.result?.fileBinary) {
                        throw new Error('Invalid file data received from Dropbox');
                    }

                    fs.writeFileSync(tempPath, fileData.result.fileBinary);

                    // Validate image stats and metadata
                    const stats = await sharp(tempPath).stats();
                    const metadata = await sharp(tempPath).metadata();
                    
                    if (!stats?.channels || stats.channels.length < 3 || 
                        !metadata?.width || !metadata?.height) {
                        throw new Error('Invalid image data');
                    }

                    // Vision API analysis with retry
                    const visionResult = await retryAnalysis(async () => {
                        const [result] = await visionClient.annotateImage({
                            image: { content: fileData.result.fileBinary.toString('base64') },
                            features: [
                                { type: 'LABEL_DETECTION', maxResults: 20 },
                                { type: 'IMAGE_PROPERTIES' },
                                { type: 'OBJECT_LOCALIZATION' },
                                { type: 'TEXT_DETECTION' }
                            ]
                        });
                        return result;
                    });

                    // Process image analysis
                    const validatedStats = tattooAnalyzer.validateStats(stats);
                    if (!validatedStats) {
                        throw new Error('Failed to validate image statistics');
                    }

                    const analysis = await tattooAnalyzer.analyzeImage({
                        stats: validatedStats,
                        visionAnalysis: visionResult,
                        metadata
                    }, entry.path_lower);

                    // Sequence detection
                    const sequences = await sequenceDetector.findRelatedImages(entry.path_lower, imageSignatures);
                    const sequenceMetrics = sequences.length > 0 ? 
                        sequenceDetector.calculateSequenceMetrics(sequences) : null;

                    // Build and store signature
                    const signature = buildImageSignature(entry, fileData, metadata, stats, visionResult, analysis, sequences, sequenceMetrics);
                    
                    // Store in memory and database
                    await storeSignature(entry.path_lower, signature);
                    
                    processedEntries.add(entry.path_lower);
                    initializationProgress.processed++;

                    // Save progress to Supabase
                    await updateProgress(processedEntries, failedEntries);

                } catch (error) {
                    logger.error(`Error processing ${entry.path_lower}:`, error);
                    failedEntries.add(entry.path_lower);
                    
                    // Track failed analysis
                    await supabase
                        .from('failed_analysis')
                        .insert({
                            path: entry.path_lower,
                            error: error.message,
                            attempted_at: new Date().toISOString()
                        });

                } finally {
                    if (fs.existsSync(tempPath)) {
                        fs.unlinkSync(tempPath);
                    }
                }
            }));

            // Log batch progress
            logger.info(`Processed ${Math.min(i + BATCH_SIZE, imageFiles.length)}/${imageFiles.length} images`);
        }

        // Finalize initialization
        isInitialized = true;
        initializationProgress.status = 'complete';
        initializationProgress.endTime = Date.now();
        
        // Log completion metrics
        logger.info('Cache initialization complete:', {
            total: imageFiles.length,
            processed: processedEntries.size,
            failed: failedEntries.size,
            duration: `${Math.round((Date.now() - startTime) / 1000)}s`
        });

        // Clear initialization progress
        await supabase
            .from('initialization_progress')
            .delete()
            .eq('id', 'latest');

    } catch (error) {
        initializationProgress.status = 'error';
        initializationProgress.lastError = error.message;
        logger.error('Initialization error:', error);
        
        // Save error state
        await updateProgress(processedEntries, failedEntries, error.message);
        
        throw error;
    }
}

// Helper functions
async function retryAnalysis(operation, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            if (attempt === maxRetries - 1) throw error;
            await new Promise(resolve => 
                setTimeout(resolve, 1000 * Math.pow(2, attempt))
            );
        }
    }
}

async function updateProgress(processed, failed, error = null) {
    return supabase
        .from('initialization_progress')
        .upsert({
            id: 'latest',
            processed_paths: Array.from(processed),
            failed_paths: Array.from(failed),
            last_error: error,
            updated_at: new Date().toISOString()
        });
}

function buildImageSignature(entry, fileData, metadata, stats, visionResult, analysis, sequences, sequenceMetrics) {
    return {
        path: entry.path_lower,
        fileName: entry.name,
        visionAnalysis: {
            labels: visionResult.labelAnnotations || [],
            objects: visionResult.localizedObjectAnnotations || [],
            colors: visionResult.imagePropertiesAnnotation?.dominantColors?.colors || [],
            text: visionResult.textAnnotations || []
        },
        tattooFeatures: analysis.stages ? {
            ...analysis.stages,
            metrics: analysis.metrics,
            patterns: analysis.patterns,
            removal: analysis.stages.during ? {
                stage: analysis.stages.during.isDuringTreatment ? 'during' :
                       analysis.stages.before?.isOriginal ? 'before' : 
                       analysis.stages.after?.isComplete ? 'after' : 'in_progress',
                progress: analysis.metrics,
                dotPattern: analysis.patterns?.dotPattern,
                fadingAnalysis: analysis.patterns?.fadingPattern
            } : null
        } : null,
        imageStats: {
            means: stats.channels.map(c => c.mean),
            stdDevs: stats.channels.map(c => c.std),
            histogram: stats.channels.map(c => c.histogram)
        },
        visualFeatures: {
            aspectRatio: metadata.width / metadata.height,
            orientation: metadata.orientation,
            format: metadata.format,
            dimensions: {
                width: metadata.width,
                height: metadata.height
            }
        },
        sequenceInfo: sequences.length > 0 ? {
            isPartOfSequence: true,
            sequenceMetrics,
            relatedImages: sequences.map(img => ({
                path: img.path,
                position: img.sequence.sequencePosition,
                timestamp: img.metadata.modified
            }))
        } : null,
        metadata: {
            size: entry.size,
            modified: entry.server_modified,
            directory: path.dirname(entry.path_lower),
            hash: crypto.createHash('md5').update(fileData.result.fileBinary).digest('hex'),
            analyzedAt: new Date().toISOString()
        }
    };
}

async function storeSignature(path, signature) {
    imageSignatures.set(path, signature);
    await supabase
        .from('image_signatures')
        .upsert({
            path,
            signature,
            analyzed_at: new Date().toISOString()
        });
}

// Helper functions for tattoo analysis
function detectBodyPlacement(labels) {
    const bodyParts = {
        arm: ['arm', 'forearm', 'bicep', 'shoulder'],
        leg: ['leg', 'thigh', 'calf', 'ankle'],
        torso: ['chest', 'back', 'torso', 'stomach'],
        head: ['neck', 'face', 'head']
    };

    for (const [area, keywords] of Object.entries(bodyParts)) {
        if (labels.some(label => 
            keywords.some(keyword => 
                label.description.toLowerCase().includes(keyword)
            )
        )) {
            return area;
        }
    }
    return 'unknown';
}

function calculateInkDensity(imageProperties) {
    if (!imageProperties?.dominantColors?.colors) return 0;
    
    return imageProperties.dominantColors.colors.reduce((sum, color) => {
        const brightness = (color.color.red + color.color.green + color.color.blue) / 3;
        return sum + ((255 - brightness) * color.pixelFraction);
    }, 0);
}

function detectTattooStyle(visionResult) {
    const styleKeywords = {
        traditional: ['traditional', 'old school', 'sailor'],
        realistic: ['realistic', 'portrait', 'photorealistic'],
        tribal: ['tribal', 'polynesian', 'maori'],
        geometric: ['geometric', 'pattern', 'mandala']
    };

    const labels = visionResult.labelAnnotations || [];
    for (const [style, keywords] of Object.entries(styleKeywords)) {
        if (labels.some(label => 
            keywords.some(keyword => 
                label.description.toLowerCase().includes(keyword)
            )
        )) {
            return style;
        }
    }
    return 'unknown';
}

// Call this when server starts

async function getQuickImageFeatures(imagePath) {
  const image = sharp(imagePath);
  const [metadata, stats] = await Promise.all([
    image.metadata(),
    image.stats()
  ]);

  const dominantColors = stats.channels.map(channel => ({
    mean: channel.mean,
    std: channel.std
  }));

  return {
    dimensions: {
      width: metadata.width,
      height: metadata.height
    },
    dominantColors,
    aspectRatio: metadata.width / metadata.height
  };
}

// Quick similarity calculation
function calculateQuickSimilarity(features1, features2) {
  let score = 0;
  
  // Compare aspect ratios
  const aspectRatioDiff = Math.abs(features1.aspectRatio - features2.aspectRatio);
  score += (1 - Math.min(aspectRatioDiff, 1)) * 0.3;

  // Compare dominant colors
  const colorSimilarity = features1.dominantColors.reduce((sum, color1, i) => {
    const color2 = features2.dominantColors[i];
    const meanDiff = Math.abs(color1.mean - color2.mean) / 255;
    const stdDiff = Math.abs(color1.std - color2.std) / 255;
    return sum + (1 - meanDiff) * 0.5 + (1 - stdDiff) * 0.5;
  }, 0) / features1.dominantColors.length;
  
  score += colorSimilarity * 0.7;

  return score;
}

async function calculateQuickSimilarity(path1, path2) {
  const [img1Stats, img2Stats] = await Promise.all([
    sharp(path1).stats(),
    sharp(path2).stats()
  ]);

  // Compare channel statistics
  let totalDiff = 0;
  for (let i = 0; i < img1Stats.channels.length; i++) {
    const chan1 = img1Stats.channels[i];
    const chan2 = img2Stats.channels[i];
    totalDiff += Math.abs(chan1.mean - chan2.mean) / 255;
  }

  return 1 - (totalDiff / img1Stats.channels.length);
}

async function calculateImageSimilarity(features1, features2) {
  if (!features1 || !features2) return 0;

  let score = 0;
  let weights = 0;

  // Compare colors
  if (features1.inkColors && features2.inkColors) {
    const colorScore = compareColors(features1.inkColors, features2.inkColors);
    score += colorScore * 0.4;
    weights += 0.4;
  }

  // Compare placement
  if (features1.placement && features2.placement) {
    if (features1.placement === features2.placement) {
      score += 0.3;
      weights += 0.3;
    }
  }

  // Compare size/density
  if (features1.detailedAnalysis && features2.detailedAnalysis) {
    const densityDiff = Math.abs(
      features1.detailedAnalysis.density - features2.detailedAnalysis.density
    );
    const densityScore = 1 - (densityDiff / 100);
    score += densityScore * 0.3;
    weights += 0.3;
  }

  return weights > 0 ? score / weights : 0;
}

// Helper function to compare color profiles
function compareColors(colors1, colors2) {
  if (!colors1 || !colors2 || !colors1.length || !colors2.length) return 0;

  const normalizedColors1 = colors1.map(normalizeColor);
  const normalizedColors2 = colors2.map(normalizeColor);

  let totalSimilarity = 0;
  let comparisons = 0;

  for (const color1 of normalizedColors1) {
    let bestMatch = 0;
    for (const color2 of normalizedColors2) {
      const similarity = 1 - (
        Math.abs(color1.r - color2.r) +
        Math.abs(color1.g - color2.g) +
        Math.abs(color1.b - color2.b)
      ) / 765; // 765 = 255 * 3
      bestMatch = Math.max(bestMatch, similarity);
    }
    totalSimilarity += bestMatch;
    comparisons++;
  }

  return totalSimilarity / comparisons;
}

// Helper function to normalize RGB color
function normalizeColor(color) {
  if (typeof color === 'string') {
    // Parse RGB string
    const matches = color.match(/\d+/g);
    return {
      r: parseInt(matches[0]),
      g: parseInt(matches[1]),
      b: parseInt(matches[2])
    };
  }
  return color;
}

// Helper function to find sequence in directory
async function findSequenceInDirectory(directory) {
  const entries = await dropboxManager.fetchDropboxEntries(directory);
  const sequenceFiles = entries.result.entries
    .filter(entry => ['.jpg', '.jpeg', '.png'].some(ext => 
      entry.path_lower.endsWith(ext)
    ))
    .sort((a, b) => new Date(a.server_modified) - new Date(b.server_modified))
    .map(entry => ({
      path: entry.path_lower,
      metadata: {
        modified: entry.server_modified,
        size: entry.size
      },
      isSequence: true
    }));

  return sequenceFiles;
}

async function getColorFingerprint(imagePath) {
  const img = sharp(imagePath);
  const { data } = await img
    .resize(8, 8, { fit: 'fill' })
    .raw()
    .toBuffer();
  
  return Array.from(data);
}

function compareFingerprints(fp1, fp2) {
  let diff = 0;
  for (let i = 0; i < fp1.length; i++) {
    diff += Math.abs(fp1[i] - fp2[i]);
  }
  return 1 - (diff / (fp1.length * 255));
}

async function calculateDetailedSimilarity(path1, path2) {
  const [stats1, stats2] = await Promise.all([
    sharp(path1).stats(),
    sharp(path2).stats()
  ]);

  const channelScores = stats1.channels.map((chan1, i) => {
    const chan2 = stats2.channels[i];
    const meanDiff = Math.abs(chan1.mean - chan2.mean) / 255;
    const stdDiff = Math.abs(chan1.std - chan2.std) / 255;
    return (1 - meanDiff) * 0.6 + (1 - stdDiff) * 0.4;
  });

  return channelScores.reduce((a, b) => a + b) / channelScores.length;
}

function compareImageProfiles(profile1, profile2) {
  let score = 0;
  let weights = 0;

  // Compare dimensions (quick)
  const aspectRatio1 = profile1.dimensions.width / profile1.dimensions.height;
  const aspectRatio2 = profile2.dimensions.width / profile2.dimensions.height;
  const aspectRatioDiff = Math.abs(aspectRatio1 - aspectRatio2);
  score += (1 - Math.min(aspectRatioDiff, 1)) * 0.3;
  weights += 0.3;

  // Compare color statistics (quick)
  const stats1 = profile1.colorStats;
  const stats2 = profile2.colorStats;
  
  for (let i = 0; i < stats1.channels.length; i++) {
    const channel1 = stats1.channels[i];
    const channel2 = stats2.channels[i];
    
    const meanDiff = Math.abs(channel1.mean - channel2.mean) / 255;
    const stdDiff = Math.abs(channel1.std - channel2.std) / 255;
    
    score += (1 - meanDiff) * 0.2;
    score += (1 - stdDiff) * 0.1;
    weights += 0.3;
  }

  return score / weights;
}

// Enhanced endpoints
app.get('/test', (req, res) => {
  console.log('Test endpoint hit');
  res.json({ ok: true, time: Date.now() });
});

app.get('/initialization-status', (req, res) => {
    res.json({
        isInitialized,
        progress: initializationProgress
    });
});


app.get('/supabase/test', async (req, res) => {
  try {
    // Test database connection
    const { data: documents, error: documentsError } = await supabase
      .from('documents')
      .select('*')
      .limit(5);

    const { data: chatHistory, error: chatHistoryError } = await supabase
      .from('chat_history')
      .select('*')
      .limit(5);

    const { data: processingQueue, error: queueError } = await supabase
      .from('processing_queue')
      .select('*')
      .limit(5);

    const { data: documentRelevance, error: relevanceError } = await supabase.rpc(
      'calculate_document_relevance',
      {
        doc_age: '7 days',
        doc_type: 'pdf',
        metadata: { hasAnalysis: true }
      }
    );

    const { data: matchedDocuments, error: matchError } = await supabase.rpc(
      'match_documents',
      {
        query_embedding: Array(1536).fill(0.1), // Mock embedding for testing
        match_threshold: 0.5,
        match_count: 5
      }
    );

    const { data: chatSearchResults, error: chatSearchError } = await supabase.rpc(
      'search_chat_history',
      {
        query_embedding: Array(1536).fill(0.1), // Mock embedding for testing
        user_id: '00000000-0000-0000-0000-000000000000', // Replace with a valid user ID
        match_threshold: 0.5,
        match_count: 5
      }
    );

    // Consolidate results
    const status = {
      connection: true,
      documents: {
        success: !documentsError,
        count: documents?.length || 0,
        error: documentsError?.message || null,
      },
      chatHistory: {
        success: !chatHistoryError,
        count: chatHistory?.length || 0,
        error: chatHistoryError?.message || null,
      },
      processingQueue: {
        success: !queueError,
        count: processingQueue?.length || 0,
        error: queueError?.message || null,
      },
      relevanceFunction: {
        success: !relevanceError,
        results: documentRelevance || [],
        error: relevanceError?.message || null,
      },
      matchFunction: {
        success: !matchError,
        results: matchedDocuments || [],
        error: matchError?.message || null,
      },
      chatSearchFunction: {
        success: !chatSearchError,
        results: chatSearchResults || [],
        error: chatSearchError?.message || null,
      },
    };

    res.json(status);
  } catch (error) {
    res.status(500).json({ error: 'Failed to test Supabase functionality', details: error.message });
  }
});


// Change the upload endpoint to be simpler first
app.post('/upload', upload.single('image'), async (req, res) => {  // Made the handler async
  const processingId = crypto.randomUUID();
  console.log('1. Upload request start:', { processingId });
  
  // Send immediate acknowledgment
  res.json({
    received: true,
    file: req.file ? {
      filename: req.file.filename,
      size: req.file.size
    } : null,
    processingId
  });
  
  if (!req.file) {
    console.log('2. No file received');
    return;
  }

  try {
    console.log('3. Processing file:', req.file.filename);

    const dropboxStatus = await dropboxManager.ensureAuth();
    console.log('4. Dropbox status:', !!dropboxStatus);

    const [visionResult] = await visionClient.labelDetection(req.file.path);
    console.log('5. Vision API analysis complete');

    // Store in documents table first
    const { data: document, error: docError } = await supabase
      .from('documents')
      .insert({
        id: processingId,
        content: JSON.stringify(visionResult.labelAnnotations),
        metadata: {
          filename: req.file.filename,
          size: req.file.size,
          mimetype: req.file.mimetype,
          originalName: req.file.originalname,
          dropboxConnected: !!dropboxStatus
        },
        document_type: 'image',
        source_type: 'upload',
        status: 'active'
      })
      .select()
      .single();

    if (docError) {
      throw new Error(`Document insert error: ${docError.message}`);
    }
    console.log('6. Stored document:', processingId);

    // Store in image_sequences table
    const { error: seqError } = await supabase
      .from('image_sequences')
      .insert({
        image_id: processingId,
        sequence_group: `upload-${Date.now()}`,
        sequence_order: 1,
        sequence_metadata: {
          analysis: visionResult.labelAnnotations,
          dropboxStatus: !!dropboxStatus,
          processed_at: new Date().toISOString()
        }
      });

    if (seqError) {
      throw new Error(`Sequence insert error: ${seqError.message}`);
    }
    console.log('7. Stored sequence data');

  } catch (error) {
    console.error('Processing error:', error);
  } finally {
    // Clean up file
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
      console.log('8. File cleaned up');
    }
  }
});

// Add a status check endpoint for verifying data
app.get('/status/:processingId', async (req, res) => {
  try {
    const { processingId } = req.params;

    // Check both tables
    const [docResult, seqResult] = await Promise.all([
      supabase
        .from('documents')
        .select('*')
        .eq('id', processingId)
        .single(),
      supabase
        .from('image_sequences')
        .select('*')
        .eq('image_id', processingId)
        .single()
    ]);

    res.json({
      status: 'ok',
      document: docResult.data,
      sequence: seqResult.data,
      errors: {
        document: docResult.error?.message,
        sequence: seqResult.error?.message
      }
    });

  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message
    });
  }
});

app.get('/test/db-config', async (req, res) => {
  try {
    // First check credentials
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_KEY) {
      return res.status(500).json({
        status: 'error',
        error: 'Missing Supabase credentials',
        config: {
          hasUrl: !!process.env.SUPABASE_URL,
          hasKey: !!process.env.SUPABASE_KEY
        }
      });
    }

    // Test connection
    const { data, error } = await supabase
      .from('processed_images')
      .select('count');

    if (error) throw error;

    res.json({
      status: 'ok',
      config: {
        hasUrl: !!process.env.SUPABASE_URL,
        hasKey: !!process.env.SUPABASE_KEY,
        urlPreview: `${process.env.SUPABASE_URL.slice(0, 20)}...`,
        keyPreview: `${process.env.SUPABASE_KEY.slice(0, 8)}...`
      },
      connection: 'successful',
      data: data
    });

  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message,
      config: {
        hasUrl: !!process.env.SUPABASE_URL,
        hasKey: !!process.env.SUPABASE_KEY
      }
    });
  }
});


// Add a direct Supabase test endpoint
app.get('/test/supabase', async (req, res) => {
  console.log('Testing Supabase connection');
  
  try {
    console.log('Supabase URL:', process.env.SUPABASE_URL);
    console.log('Supabase Key exists:', !!process.env.SUPABASE_KEY);
    
    // Try a simple query
    const { data, error } = await supabase
      .from('documents')
      .select('count');
      
    console.log('Query result:', { data, error });

    if (error) {
      throw error;
    }

    res.json({
      status: 'connected',
      config: {
        url: process.env.SUPABASE_URL,
        hasKey: !!process.env.SUPABASE_KEY
      },
      data
    });
  } catch (error) {
    console.error('Supabase test error:', error);
    
    res.status(500).json({
      status: 'error',
      error: error.message,
      config: {
        url: process.env.SUPABASE_URL,
        hasKey: !!process.env.SUPABASE_KEY
      }
    });
  }
});

// Service status check
app.get('/status/check/services', async (req, res) => {
  try {
    const [dropboxStatus, visionStatus, supabaseStatus] = await Promise.all([
      dropboxManager.ensureAuth(),
      visionClient.labelDetection(Buffer.from('test')).then(() => true).catch(() => false),
      supabase.from('processed_images').select('count').then(() => true).catch(() => false)
    ]);

    res.json({
      status: 'ok',
      services: {
        dropbox: !!dropboxStatus,
        vision: !!visionStatus,
        supabase: !!supabaseStatus
      },
      timestamp: Date.now()
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message,
      timestamp: Date.now()
    });
  }
});

// Modified status endpoint
app.get('/status/check', async (req, res) => {
  try {
    const [dropboxStatus, visionStatus] = await Promise.all([
      dropboxManager.ensureAuth(),
      visionClient.labelDetection(Buffer.from('test'))
        .then(() => true)
        .catch(() => false)
    ]);

    res.json({
      status: 'ok',
      services: {
        dropbox: !!dropboxStatus,
        vision: !!visionStatus
      },
      timestamp: Date.now()
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message,
      timestamp: Date.now()
    });
  }
});

// Simplified processing status endpoint

// Simple status check
app.get('/status/test', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: Date.now()
  });
});



// Simplify the chat endpoint too
app.post('/chat', async (req, res) => {
  try {
    const { message, userId } = req.body;
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const userUUID = generateUUID(userId || 'default-user');
    console.log('1. Chat request received:', { userId, userUUID });

    // Get or create assistant first
    const assistant = await getOrCreateAssistant();
    console.log('2. Assistant ID:', assistant.id);

    // Get or create thread
    const threadId = await getOrCreateThread(userId || 'default-user');
    console.log('3. Thread ID:', threadId);

    // Store chat in Supabase
    const { error: chatError } = await supabase
      .from('chat_history')
      .insert({
        user_id: userUUID,
        thread_id: threadId,
        message_type: 'user',
        content: message
      });

    if (chatError) {
      console.error('Chat storage error:', chatError);
    }

    // Add message to thread
    await openai.beta.threads.messages.create(threadId, {
      role: "user",
      content: message
    });
    console.log('4. Added message to thread');

    // Run assistant
    const run = await openai.beta.threads.runs.create(threadId, {
      assistant_id: assistant.id
    });
    console.log('5. Started assistant run');

    // Wait for completion with timeout
    const startTime = Date.now();
    const timeout = 30000; // 30 second timeout

    let assistantResponse = null;
    
    while (Date.now() - startTime < timeout) {
      const status = await openai.beta.threads.runs.retrieve(threadId, run.id);
      console.log('6. Run status:', status.status);
      
      if (status.status === 'completed') {
        const messages = await openai.beta.threads.messages.list(threadId);
        assistantResponse = messages.data[0].content[0].text.value;
        console.log('7. Got assistant response');
        
        // Store assistant response in Supabase
        await supabase
          .from('chat_history')
          .insert({
            user_id: userUUID,
            thread_id: threadId,
            message_type: 'assistant',
            content: assistantResponse
          });
        
        break;
      } else if (status.status === 'failed') {
        throw new Error('Assistant run failed');
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    if (!assistantResponse) {
      throw new Error('Assistant response timeout');
    }

    res.json({
      response: assistantResponse,
      threadId,
      assistantId: assistant.id
    });

  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ 
      error: error.message,
      errorType: error.constructor.name
    });
  }
});

// Modified chat history endpoint
app.get('/chat/history/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const userUUID = generateUUID(userId || 'default-user');
    
    const { data, error } = await supabase
      .from('chat_history')
      .select('*')
      .eq('user_id', userUUID)
      .order('created_at', { ascending: true });

    if (error) throw error;

    res.json({
      history: data
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Add chat history endpoint
app.get('/chat/history/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { data, error } = await supabase
      .from('chat_history')
      .select('*')
      .eq('user_id', userId || '00000000-0000-0000-0000-000000000000')
      .order('created_at', { ascending: true });

    if (error) throw error;

    res.json({
      history: data
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/analyze', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    const [result] = await visionClient.labelDetection(req.file.path);
    res.json({ labels: result.labelAnnotations });
  } catch (error) {
    res.status(500).json({ error: error.message });
  } finally {
    fs.unlinkSync(req.file.path); // Cleanup file after processing
  }
});

// Add a simple search endpoint
app.get('/search', async (req, res) => {
  console.log('Search request received');
  
  try {
    const { query } = req.query;
    console.log('Search query:', query);
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    res.json({ 
      success: true,
      query: query,
      message: 'Search endpoint reached'
    });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/dropbox/auth', (req, res) => {
  try {
    const authUrl = dropboxManager.getAuthUrl();
    res.redirect(authUrl);
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate Dropbox auth URL' });
  }
});

app.get('/dropbox/callback', async (req, res) => {
  try {
    const { code } = req.query;
    if (!code) {
      return res.status(400).json({ error: 'Authorization code missing' });
    }
    await dropboxManager.handleCallback(code);
    res.send('Dropbox authentication successful!');
  } catch (error) {
    res.status(500).json({ error: 'Authentication failed' });
  }
});

app.get('/dropbox/status', async (req, res) => {
  try {
    const dropboxStatus = await dropboxManager.ensureAuth();
    res.json({
      status: 'ok',
      dropbox: {
        connected: !!dropboxStatus,
        lastSync: dropboxManager.lastSyncTime || null
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/dropbox/file', async (req, res) => {
  try {
    const { path } = req.query;
    if (!path) {
      return res.status(400).json({ error: 'Path parameter is required' });
    }

    const fileData = await dropboxManager.downloadFile(path);
    res.json({
      success: true,
      file: fileData.result,
      metadata: {
        path: path,
        modified: fileData.result.metadata?.server_modified,
        size: fileData.result.metadata?.size
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Search files
app.get('/dropbox/search', async (req, res) => {
  try {
    const { query } = req.query;
    if (!query) {
      return res.status(400).json({ error: 'Search query is required' });
    }

    const entries = await dropboxManager.fetchDropboxEntries('');
    const matchingFiles = entries.result.entries.filter(entry => 
      entry.path_lower.includes(query.toLowerCase())
    );

    res.json({
      success: true,
      query,
      results: matchingFiles,
      count: matchingFiles.length
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Image analysis endpoint
app.post('/analyze/image', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file provided' });
  }

  try {
    // Vision API analysis
    const [visionResult] = await visionClient.labelDetection(req.file.path);
    
    // Process image
    const imageAnalysis = await imageProcessor.processImage({
      path: req.file.path,
      metadata: {
        filename: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype
      }
    });

    res.json({
      success: true,
      analysis: {
        labels: visionResult.labelAnnotations,
        imageFeatures: imageAnalysis,
        metadata: {
          filename: req.file.originalname,
          size: req.file.size,
          analyzed: new Date().toISOString()
        }
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  } finally {
    // Clean up uploaded file
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
  }
});

// Find similar images
app.post('/analyze/similar', async (req, res) => {
  try {
    const { path } = req.body;
    if (!path) {
      return res.status(400).json({ error: 'Image path is required' });
    }

    // Get all images
    const entries = await dropboxManager.fetchDropboxEntries('');
    const imageFiles = entries.result.entries.filter(entry => 
      ['.jpg', '.jpeg', '.png', '.gif'].some(ext => 
        entry.path_lower.endsWith(ext)
      )
    );

    // Group by directory for potential sequences
    const groupedFiles = imageFiles.reduce((acc, file) => {
      const dir = file.path_lower.split('/').slice(0, -1).join('/');
      if (!acc[dir]) acc[dir] = [];
      acc[dir].push(file);
      return acc;
    }, {});

    // Find sequences
    const sequences = Object.values(groupedFiles)
      .filter(group => group.length > 1)
      .map(group => ({
        directory: group[0].path_lower.split('/').slice(0, -1).join('/'),
        files: group.sort((a, b) => 
          new Date(a.server_modified) - new Date(b.server_modified)
        ),
        count: group.length
      }));

    res.json({
      success: true,
      path,
      similar: {
        totalImages: imageFiles.length,
        sequences: sequences,
        sequenceCount: sequences.length
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Directory listing
app.get('/dropbox/directory', async (req, res) => {
  try {
    const { path = '' } = req.query;
    const entries = await dropboxManager.fetchDropboxEntries(path);
    
    // Organize entries by type
    const files = entries.result.entries.filter(entry => entry['.tag'] === 'file');
    const folders = entries.result.entries.filter(entry => entry['.tag'] === 'folder');
    
    res.json({
      success: true,
      path,
      contents: {
        files: files,
        folders: folders,
        totalFiles: files.length,
        totalFolders: folders.length
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Enhanced status endpoint
app.get('/status/full', async (req, res) => {
  try {
    const [dropboxStatus, visionStatus] = await Promise.all([
      dropboxManager.ensureAuth(),
      visionClient.labelDetection(Buffer.from('test'))
        .then(() => true)
        .catch(() => false)
    ]);

    const entries = await dropboxManager.fetchDropboxEntries('');
    const imageCount = entries.result.entries.filter(entry => 
      ['.jpg', '.jpeg', '.png', '.gif'].some(ext => 
        entry.path_lower.endsWith(ext)
      )
    ).length;

    res.json({
      success: true,
      timestamp: new Date().toISOString(),
      services: {
        dropbox: {
          connected: !!dropboxStatus,
          fileCount: entries.result.entries.length,
          imageCount
        },
        vision: {
          available: visionStatus
        }
      },
      system: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/dropbox/sync', async (req, res) => {
  try {
    const status = await dropboxManager.ensureAuth();
    if (!status) {
      throw new Error('Dropbox not authenticated');
    }
    
    const entries = await dropboxManager.fetchDropboxEntries('');
    res.json({
      success: true,
      filesFound: entries.result.entries.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Add these endpoints after your existing working endpoints

// Tattoo Removal Sequence Analysis
app.post('/search/visual', upload.single('image'), async (req, res) => {
    const startTime = Date.now();
    const tempFiles = [];
    let searchPath;

    try {
        if (!isInitialized) {
            return res.status(503).json({ 
                error: 'Server is still initializing image cache',
                progress: initializationProgress
            });
        }

        // Get search image
        if (req.file) {
            searchPath = req.file.path;
        } else if (req.body.imagePath) {
            const fileData = await dropboxManager.downloadFile(req.body.imagePath);
            searchPath = path.join('uploads', `temp_${Date.now()}_search.jpg`);
            fs.writeFileSync(searchPath, fileData.result.fileBinary);
            tempFiles.push(searchPath);
        } else {
            return res.status(400).json({ error: 'No image provided' });
        }

        // Get search image signature
        const searchImage = sharp(searchPath);
        const [searchStats, searchMeta] = await Promise.all([
            searchImage.stats(),
            searchImage.metadata()
        ]);

        const searchSignature = {
            aspectRatio: searchMeta.width / searchMeta.height,
            means: searchStats.channels.map(c => c.mean),
            stdDevs: searchStats.channels.map(c => c.stdev)
        };

        // Compare with cached signatures using lower threshold
        const matches = Array.from(imageSignatures.values())
            .map(sig => ({
                ...sig,
                similarity: compareSignatures(searchSignature, sig)
            }))
            .filter(match => match.similarity > 0.6) // Lower threshold
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 20); // Get more matches

        // Calculate directory matches for sequences
        if (matches.length > 0) {
            const directories = new Map();
            matches.forEach(match => {
                const dir = path.dirname(match.path);
                if (!directories.has(dir)) {
                    directories.set(dir, []);
                }
                directories.get(dir).push(match);
            });

            // Add sequence information
            const enhancedMatches = matches.map(match => {
                const dir = path.dirname(match.path);
                const sequence = directories.get(dir);
                return {
                    ...match,
                    isPartOfSequence: sequence.length > 1,
                    sequenceInfo: sequence.length > 1 ? {
                        total: sequence.length,
                        position: sequence.findIndex(m => m.path === match.path) + 1,
                        directory: dir
                    } : null
                };
            });

            const processingTime = Date.now() - startTime;
            console.log(`Search completed in ${processingTime}ms`);

            res.json({
                success: true,
                matches: enhancedMatches,
                metadata: {
                    processingTime,
                    totalImages: imageSignatures.size,
                    matchesFound: matches.length,
                    searchedPath: searchPath,
                    searchSignature: {
                        means: searchSignature.means,
                        aspectRatio: searchSignature.aspectRatio
                    }
                }
            });
        } else {
            res.json({
                success: true,
                matches: [],
                metadata: {
                    processingTime: Date.now() - startTime,
                    totalImages: imageSignatures.size,
                    matchesFound: 0,
                    searchedPath: searchPath,
                    searchSignature: {
                        means: searchSignature.means,
                        aspectRatio: searchSignature.aspectRatio
                    }
                }
            });
        }

    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ error: error.message });
    } finally {
        tempFiles.forEach(file => {
            if (fs.existsSync(file)) fs.unlinkSync(file);
        });
    }
});

// Improved comparison function
function compareSignatures(sig1, sig2) {
    // Color similarity (70% weight)
    const colorScore = sig1.means.reduce((score, mean1, i) => {
        const mean2 = sig2.means[i] || 0;
        const diff = Math.abs(mean1 - mean2) / 255;
        return score + (1 - diff);
    }, 0) / sig1.means.length;

    // Aspect ratio similarity (30% weight)
    const arScore = sig1.aspectRatio && sig2.aspectRatio ? 
        1 - Math.min(Math.abs(sig1.aspectRatio - sig2.aspectRatio) / Math.max(sig1.aspectRatio, sig2.aspectRatio), 1) : 
        0;

    // Combined score
    return (colorScore * 0.7) + (arScore * 0.3);
}

// Fast comparison function
function compareFast(sig1, sig2) {
    // Compare aspect ratios (20% weight)
    const arDiff = Math.abs(sig1.aspectRatio - sig2.aspectRatio);
    const arScore = 1 - Math.min(arDiff, 1);

    // Compare color means (80% weight)
    const colorScore = sig1.means.reduce((score, mean1, i) => {
        const mean2 = sig2.means[i];
        return score + (1 - Math.abs(mean1 - mean2) / 255);
    }, 0) / sig1.means.length;

    return (arScore * 0.2) + (colorScore * 0.8);
}

// Fast feature comparison
function compareFeatures(feat1, feat2) {
  // Compare aspect ratios (20% weight)
  const arDiff = Math.abs(feat1.aspectRatio - feat2.aspectRatio);
  const arScore = 1 - Math.min(arDiff, 1);

  // Compare color profiles (80% weight)
  const colorScore = feat1.colorProfile.reduce((score, color1, i) => {
    const color2 = feat2.colorProfile[i];
    return score + (1 - Math.abs(color1 - color2) / 255);
  }, 0) / feat1.colorProfile.length;

  return (arScore * 0.2) + (colorScore * 0.8);
}

async function extractImageFeatures(imagePath) {
  const img = sharp(imagePath);
  const [stats, meta] = await Promise.all([
    img.stats(),
    img.metadata()
  ]);

  return {
    aspectRatio: meta.width / meta.height,
    size: meta.size,
    channels: stats.channels.map(c => ({
      mean: c.mean,
      std: c.std
    })),
    timestamp: Date.now()
  };
}

function calculateSimilarity(features1, features2) {
  // Compare aspect ratios (20%)
  const arDiff = Math.abs(features1.aspectRatio - features2.aspectRatio);
  const arScore = 1 - Math.min(arDiff, 1);

  // Compare channel statistics (80%)
  const channelScore = features1.channels.reduce((score, channel1, i) => {
    const channel2 = features2.channels[i];
    const meanDiff = Math.abs(channel1.mean - channel2.mean) / 255;
    const stdDiff = Math.abs(channel1.std - channel2.std) / 255;
    return score + ((1 - meanDiff) * 0.6 + (1 - stdDiff) * 0.4);
  }, 0) / features1.channels.length;

  return (arScore * 0.2) + (channelScore * 0.8);
}


async function getImageSignature(imagePath) {
  const img = sharp(imagePath);
  const [stats, metadata] = await Promise.all([
    img.stats(),
    img.metadata()
  ]);

  return {
    aspectRatio: metadata.width / metadata.height,
    channels: stats.channels.map(c => ({
      mean: c.mean,
      std: c.std,
      dominant: c.dominant
    })),
    size: metadata.size,
    format: metadata.format
  };
}


function compareImages(profile1, profile2) {
  // Compare color channels
  const channelSimilarity = profile1.channels.reduce((sum, channel1, i) => {
    const channel2 = profile2.channels[i];
    const meanDiff = Math.abs(channel1.mean - channel2.mean) / 255;
    const stdDiff = Math.abs(channel1.std - channel2.std) / 255;
    return sum + (1 - meanDiff) * 0.6 + (1 - stdDiff) * 0.4;
  }, 0) / profile1.channels.length;

  // Compare aspect ratios
  const arDiff = Math.abs(profile1.aspectRatio - profile2.aspectRatio);
  const arSimilarity = 1 - Math.min(arDiff / 0.5, 1);

  // Weight channels more heavily than aspect ratio
  return channelSimilarity * 0.8 + arSimilarity * 0.2;
}


// Modified sequence analysis endpoint for full processing
app.post('/analyze/sequence', async (req, res) => {
  try {
    console.log('Starting full sequence analysis for path:', req.body.directoryPath);
    const { directoryPath } = req.body;
    if (!directoryPath) {
      return res.status(400).json({ error: 'Directory path is required' });
    }

    // Get all files in directory
    const entries = await dropboxManager.fetchDropboxEntries(directoryPath);
    console.log('Found entries:', entries.result.entries.length);

    const imageFiles = entries.result.entries
      .filter(entry => ['.jpg', '.jpeg', '.png'].some(ext => 
        entry.path_lower.endsWith(ext)
      ))
      .sort((a, b) => new Date(a.server_modified) - new Date(b.server_modified));
    
    console.log('Processing all images:', imageFiles.length);

    // Process all images in sequence
    const processedImages = [];

    for (const file of imageFiles) {
      const fileData = await dropboxManager.downloadFile(file.path_lower);
      const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path_lower)}`);
      
      try {
        fs.writeFileSync(tempPath, fileData.result.fileBinary);
        const analysis = await imageProcessor.processImage({ path: tempPath });
        
        processedImages.push({
          path: file.path_lower,
          timestamp: file.server_modified,
          analysis: analysis,
          metadata: {
            size: file.size,
            name: file.name
          }
        });
      } finally {
        if (fs.existsSync(tempPath)) {
          fs.unlinkSync(tempPath);
        }
      }
    }

    // Calculate progression metrics for all images
    const progressionMetrics = processedImages.map((curr, index) => {
      if (index === 0) return null;
      const prev = processedImages[index - 1];
      
      return {
        timeFrame: {
          from: prev.timestamp,
          to: curr.timestamp,
          daysElapsed: Math.floor((new Date(curr.timestamp) - new Date(prev.timestamp)) / (1000 * 60 * 60 * 24))
        },
        changes: {
          colorFading: calculateColorChange(prev.analysis, curr.analysis),
          densityReduction: calculateDensityChange(prev.analysis, curr.analysis)
        }
      };
    }).filter(Boolean);

    res.json({
      success: true,
      sequence: {
        totalImages: processedImages.length,
        timespan: {
          start: processedImages[0]?.timestamp,
          end: processedImages[processedImages.length - 1]?.timestamp,
          totalDays: processedImages.length > 1 ? 
            Math.floor((new Date(processedImages[processedImages.length - 1].timestamp) - 
                       new Date(processedImages[0].timestamp)) / (1000 * 60 * 60 * 24)) : 0
        },
        images: processedImages,
        progression: progressionMetrics,
        overallProgress: calculateOverallProgress(processedImages)
      }
    });
  } catch (error) {
    console.error('Sequence analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Advanced Tattoo Analysis
app.post('/analyze/tattoo', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image provided' });
  }

  try {
    // Vision API analysis
    const [visionResult] = await visionClient.labelDetection(req.file.path);
    
    // Specific tattoo analysis
    const analysis = await imageProcessor.processImage({ 
      path: req.file.path,
      type: 'tattoo'
    });

    // Enhance with tattoo-specific metrics
    const enhancedAnalysis = {
      ...analysis,
      tattooMetrics: {
        inkDensity: calculateInkDensity(analysis),
        colorDistribution: analyzeColorDistribution(analysis),
        edgeDefinition: calculateEdgeDefinition(analysis),
        skinCondition: analyzeSkinCondition(analysis)
      },
      treatmentRecommendations: generateTreatmentRecommendations(analysis)
    };

    res.json({
      success: true,
      analysis: enhancedAnalysis,
      visionAnalysis: visionResult.labelAnnotations,
      recommendations: {
        nextSteps: suggestNextSteps(enhancedAnalysis),
        expectedProgress: predictProgress(enhancedAnalysis)
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  } finally {
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
  }
});


// Batch Processing
app.post('/process/batch', async (req, res) => {
  try {
    const { paths } = req.body;
    if (!Array.isArray(paths) || paths.length === 0) {
      return res.status(400).json({ error: 'Array of paths is required' });
    }

    const results = await Promise.all(paths.map(async (path) => {
      try {
        const fileData = await dropboxManager.downloadFile(path);
        const tempPath = `uploads/temp_${Date.now()}_${crypto.randomUUID()}.jpg`;
        fs.writeFileSync(tempPath, fileData.result.fileBinary);

        const analysis = await imageProcessor.processImage({ path: tempPath });
        fs.unlinkSync(tempPath);

        return {
          path,
          success: true,
          analysis
        };
      } catch (error) {
        return {
          path,
          success: false,
          error: error.message
        };
      }
    }));

    res.json({
      success: true,
      results,
      summary: {
        total: paths.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Helper functions
function calculateColorChange(prev, curr) {
  // Implementation details for color change calculation
  return 0.5; // Placeholder
}

function calculateDensityChange(prev, curr) {
  // Implementation details for density change calculation
  return 0.5; // Placeholder
}

function calculateOverallProgress(images) {
  // Implementation details for overall progress calculation
  return {
    percentageComplete: 75, // Placeholder
    estimatedSessionsRemaining: 2 // Placeholder
  };
}

function calculateInkDensity(analysis) {
  // Implementation details for ink density calculation
  return 0.5; // Placeholder
}

function analyzeColorDistribution(analysis) {
  // Implementation details for color distribution analysis
  return {
    primary: 'black',
    secondary: 'blue',
    distribution: { black: 0.7, blue: 0.3 }
  };
}

function calculateEdgeDefinition(analysis) {
  // Implementation details for edge definition calculation
  return 0.8; // Placeholder
}

function analyzeSkinCondition(analysis) {
  // Implementation details for skin condition analysis
  return {
    irritation: 'low',
    healing: 'good'
  };
}

function generateTreatmentRecommendations(analysis) {
  // Implementation details for treatment recommendations
  return {
    nextSession: '2 weeks',
    focus: 'darker areas'
  };
}

function suggestNextSteps(analysis) {
  // Implementation details for next steps suggestions
  return [
    'Continue with standard intervals',
    'Focus on outlined areas'
  ];
}

function predictProgress(analysis) {
  // Implementation details for progress prediction
  return {
    estimatedSessions: 4,
    expectedDuration: '6 months'
  };
}

function calculateImageSimilarity(img1, img2) {
  // Implementation details for image similarity calculation
  return 0.8; // Placeholder
}

function findImageSequences(images) {
  // Implementation details for finding image sequences
  return [{
    id: crypto.randomUUID(),
    images: images.slice(0, 3),
    timeline: 'monthly'
  }];
}

const httpsOptions = {
  key: fs.readFileSync(process.env.SSL_KEY_PATH),
  cert: fs.readFileSync(process.env.SSL_CERT_PATH),
  secureProtocol: 'TLSv1_2_method',
  rejectUnauthorized: false,
  requestCert: false,
  agent: false,
};

// Start Servers
const httpServer = http.createServer(app);
const httpsServer = https.createServer(httpsOptions, app);

async function startServer() {
    try {
        // Start HTTPS server
        httpsServer.listen(HTTPS_PORT, '0.0.0.0', () => {
            logger.info(`HTTPS server running on port ${HTTPS_PORT}`);
        });

        // Start HTTP server and initialize cache
        httpServer.listen(HTTP_PORT, '0.0.0.0', async () => {
            logger.info(`HTTP server running on port ${HTTP_PORT}`);
            
            try {
                await initializeImageCache();
                logger.info('Cache initialization completed');
            } catch (error) {
                logger.error('Cache initialization failed:', error);
            }
        });

    } catch (error) {
        logger.error('Server startup failed:', error);
        process.exit(1);
    }
}

// Start the server if this file is run directly
if (require.main === module) {
    startServer().catch(error => {
        logger.error('Failed to start server:', error);
        process.exit(1);
    });
}

module.exports = {
    app,
    httpServer,
    httpsServer,
    startServer,
    imageCacheManager,   
 sequenceDetector
};
