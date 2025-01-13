const express = require('express');
const multer = require('multer');
const path = require('path');
const http = require('http');
const https = require('https');
const sharp = require('sharp');
const { imageHash } = require('image-hash');
const ImageComparison = require('./imageComparison');
const { ImageSignatureGenerator } = require('./imageProcessor');
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
const Redis = require('redis');
const { promisify } = require('util');
const AsyncLock = require('async-lock');
const imageHashAsync = promisify(imageHash);
const signatureStore = require('./imageSignatureStore');
const EnhancedVisualSearch = require('./enhanced-search');
const OptimizedImageSearch = require('./optimized-search');
const EnhancedSignatureStore = require('./enhanced-store');
const { ImageOptimizer } = require('./performance-optimizations');
const ImageValidationSystem = require('./image-validation-system');
const EnhancedValidation = require('./enhanced-validation');
const signatureSystem = require('./signatureSystem');


let pixelmatch;
(async () => {
    pixelmatch = (await import('pixelmatch')).default;
})();

require('dotenv').config();


// Configuration validation
console.log('Environment check:', {
    SUPABASE_URL: process.env.SUPABASE_URL,
    NODE_ENV: process.env.NODE_ENV
});

// Core Constants
const HEALTH_CHECK_INTERVAL = 60 * 60 * 1000; // 1 hour
const HTTP_PORT = process.env.HTTP_PORT || 4000;
const HTTPS_PORT = process.env.HTTPS_PORT || 4001;
const MEMORY_CHECK_INTERVAL = 60000; // 1 minute
const TIMEOUT_MS = 10000; // 10 seconds timeout
const DROPBOX_TIMEOUT_MS = 15000; // 15 seconds for Dropbox operations
const IMAGE_BATCH_SIZE = 500;
const AUTH_TIMEOUT = 1000 * 60 * 5; // 5 minutes
const BATCH_SIZE = 50; // Process 50 signatures at a time
const CONCURRENT_LIMIT = 5; // Only 5 concurrent Dropbox downloads
const RATE_LIMIT_DELAY = 200; // 100ms between operations
const processedFiles = new Set();

// Analysis Thresholds
// Add these new constants at the top of server.js
const SEARCH_MODES = {
    EXACT: 'exact',
    SCREENSHOT: 'screenshot',
    SIMILAR: 'similar',
    AUTO: 'auto'
};

const SCREENSHOT_DETECTION = {
    MIN_BORDER: 5,
    MAX_BORDER_COLOR_VARIANCE: 10,
    TYPICAL_SCREENSHOT_RATIOS: [
        16/9, 4/3, 21/9, 
        9/16, // Mobile screenshots
        1 // Square screenshots
    ],
    TOLERANCE: 0.1
};



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

// Advanced comparison settings
const ADVANCED_SETTINGS = {
    HASH_SIZES: [16, 32, 64],
    ROTATIONS: [0, 90, 180, 270],
    SCALES: [0.5, 1.0, 2.0],
    COLOR_SPACES: ['srgb', 'lab', 'hsv'],
    THRESHOLDS: {
        EXACT_MATCH: 0.98,
        HIGH_CONFIDENCE: 0.95,
        MEDIUM_CONFIDENCE: 0.85,
        LOW_CONFIDENCE: 0.70,
        MIN_VALIDATION_SCORE: 0.90,
        SEQUENCE_SIMILARITY: 0.85,
        COLOR_TOLERANCE: 0.1,
        ASPECT_RATIO_TOLERANCE: 0.05
    }
};

// Add this too for comprehensive matching
const COMPARISON_WEIGHTS = {
    PERCEPTUAL_HASH: 0.25,
    COLOR_METRICS: 0.25,
    EDGE_SIGNATURE: 0.20,
    STRUCTURAL: 0.15,
    METADATA: 0.15
};

// Cache Settings
const CACHE_SETTINGS = {
    AUTH_REFRESH_INTERVAL: 4 * 60 * 1000,
    MAX_CONCURRENT_BATCHES: 3,
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY_BASE: 2000,
    BATCH_DELAY: 500,
    CACHE_TTL: 24 * 60 * 60 * 1000,
    SUPPORTED_FORMATS: ['.jpg', '.jpeg', '.png', '.webp'],
    MAX_FILE_SIZE: 10 * 1024 * 1024
};

// Analysis Settings
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

// Initialize state variables
let allImageFiles = null;
let lastAuthTime = null;
let isInitialized = false;
const featureCache = new Map();
const imageMetadataCache = new Map();
const imageSignatureCache = new Map();
const analysisCache = new Map();
const imageSignatures = new Map();
const lock = new AsyncLock();
const dropboxCache = new Map();

// Initialize Progress Tracking
let initializationProgress = {
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

// Initialize Express app
const app = express();

// Initialize Redis
const redis = Redis.createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Promisify Redis methods
const redisGet = promisify(redis.get).bind(redis);
const redisSet = promisify(redis.set).bind(redis);

// Initialize Vision client
const visionClient = new vision.ImageAnnotatorClient({
    keyFilename: path.join(__dirname, process.env.GOOGLE_APPLICATION_CREDENTIALS),
});

// Initialize Supabase
console.log('Supabase Configuration:', {
    hasUrl: !!process.env.SUPABASE_URL,
    hasKey: !!process.env.SUPABASE_KEY,
    url: process.env.SUPABASE_URL,
    keyPreview: process.env.SUPABASE_KEY ? `${process.env.SUPABASE_KEY.slice(0, 8)}...` : 'missing'
});

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

// Initialize OpenAI
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    organization: process.env.ORGANIZATION_ID,
    maxRetries: 3,
    timeout: 30000
});



// Active threads management
const activeThreads = new Map();

// Set up upload directory
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer for file uploads
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
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only images are allowed'));
        }
    },
});

// Configure rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 500, // Limit each IP to 500 requests per windowMs
    message: { error: 'Too many requests, please try again later.' },
});

// Apply middleware
app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(limiter);

// Error handling middleware
app.use((err, req, res, next) => {
    logger.error('Unhandled error:', { error: err.message, stack: err.stack });
    res.status(500).json({ error: 'Internal server error' });
});

// Process Uncaught Exceptions and Rejections
process.on('uncaughtException', (error) => logger.error('Uncaught Exception:', error));
process.on('unhandledRejection', (reason) => logger.error('Unhandled Rejection:', reason));

// Environment validation

function logSignatureDetails(signature, path) {
    logger.info('Generated signature details:', {
        service: "tatt2awai-bot",
        path,
        signatureId: signature.signatureId || 'none',
        dimensions: signature.metadata?.dimensions || 'none',
        hasPerceptualHashes: !!signature.perceptualHashes,
        hasEdgeSignature: !!signature.edgeSignature,
        features: {
            colorMetrics: !!signature.colorMetrics,
            binarySignature: !!signature.binarySignature,
            aspectRatio: signature.metadata?.dimensions?.aspectRatio
        },
        timestamp: new Date().toISOString()
    });
}

function logCacheStats() {
    logger.info('Cache statistics:', {
        service: "tatt2awai-bot",
        totalSignatures: imageSignatures.size,
        memoryUsage: process.memoryUsage().heapUsed / (1024 * 1024),
        timestamp: new Date().toISOString(),
        signatureTypes: {
            withFeatures: Array.from(imageSignatures.values())
                .filter(s => s.features).length,
            withColorMetrics: Array.from(imageSignatures.values())
                .filter(s => s.colorMetrics).length
        }
    });
}


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
// Core Analysis Utilities
const colorAnalysis = {
    calculateIntensity(color) {
        return (color.red + color.green + color.blue) / 3;
    },

    calculateSaturation(color) {
        const max = Math.max(color.red, color.green, color.blue);
        const min = Math.min(color.red, color.green, color.blue);
        return max === 0 ? 0 : (max - min) / max;
    },

    compareColors(color1, color2) {
        const distance = Math.sqrt(
            Math.pow(color1.red - color2.red, 2) +
            Math.pow(color1.green - color2.green, 2) +
            Math.pow(color1.blue - color2.blue, 2)
        );
        return 1 - (distance / (Math.sqrt(3) * 255));
    },

    analyzePalette(colors) {
        return colors.map(color => ({
            rgb: {
                red: Math.round(color.color.red),
                green: Math.round(color.color.green),
                blue: Math.round(color.color.blue)
            },
            intensity: this.calculateIntensity(color.color),
            saturation: this.calculateSaturation(color.color),
            prominence: color.pixelFraction
        }));
    }
};

// Image Quality Assessment
const qualityAssessment = {
    async assessQuality(buffer) {
        const image = sharp(buffer);
        const metadata = await image.metadata();
        const stats = await image.stats();

        return {
            dimensions: {
                width: metadata.width,
                height: metadata.height,
                aspectRatio: metadata.width / metadata.height
            },
            quality: {
                format: metadata.format,
                space: metadata.space,
                depth: metadata.depth,
                density: metadata.density
            },
            statistics: {
                contrast: this.calculateContrast(stats),
                brightness: this.calculateBrightness(stats),
                sharpness: await this.estimateSharpness(image)
            }
        };
    },

    calculateContrast(stats) {
        return stats.channels.reduce((sum, channel) => 
            sum + (channel.max - channel.min) / 255, 0) / stats.channels.length;
    },

    calculateBrightness(stats) {
        return stats.channels.reduce((sum, channel) => 
            sum + channel.mean / 255, 0) / stats.channels.length;
    },

    async estimateSharpness(image) {
        const edgeImage = await image
            .greyscale()
            .convolve({
                width: 3,
                height: 3,
                kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
            })
            .toBuffer();

        const stats = await sharp(edgeImage).stats();
        return stats.channels[0].mean / 255;
    }
};

// Quick Feature Comparison
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

// Helper Functions
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

function logAnalysisDetails(analysis, path) {
    logger.info('Detailed analysis results:', {
        path,
        hasVisionAnalysis: !!analysis.visionAnalysis,
        hasTattooFeatures: !!analysis.tattooFeatures,
        hasColorAnalysis: !!analysis.colorAnalysis,
        hasQualityMetrics: !!analysis.imageQuality,
        processingStages: analysis.metadata?.processingStages || {}
    });
}

function logSequenceDetails(sequence, path) {
    logger.info('Sequence details:', {
        path,
        isPartOfSequence: sequence.length > 0,
        sequenceLength: sequence.length,
        timespan: sequence.length > 0 ? {
            start: sequence[0].metadata.modified,
            end: sequence[sequence.length - 1].metadata.modified
        } : null
    });
}

async function generateFileHash(filePath) {
    const content = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(content).digest('hex');
}

async function processImage(file) {
    let optimizedBuffer = null;
    let tempPath = null;
    let processingStage = 'validation';
    let startTime = Date.now();

    try {
        if (!file) {
            throw new Error('No file provided');
        }

        if (file.buffer) {
            tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path || 'temp.jpg')}`);
            fs.writeFileSync(tempPath, file.buffer);
            file.path = tempPath;
        }

        const cacheKey = await generateFileHash(file.path);
        
        const { data: cachedAnalysis } = await supabase
            .from('image_analysis')
            .select('*')
            .eq('path', file.path)
            .single();

        if (cachedAnalysis && 
            (Date.now() - new Date(cachedAnalysis.analyzed_at).getTime()) < CACHE_SETTINGS.CACHE_TTL) {
            logger.info('Returning cached analysis', { path: file.path });
            return cachedAnalysis.analysis;
        }

        processingStage = 'optimization';
        optimizedBuffer = await imageProcessor.optimizeImage(file.path);

        processingStage = 'vision-api';
        const [visionResult] = await visionClient.annotateImage({
            image: { content: optimizedBuffer.toString('base64') },
            features: ANALYSIS_SETTINGS.VISION_API.FEATURES
        });

        processingStage = 'feature-analysis';
        const tattooFeatures = await imageProcessor.analyzeTattooFeatures(visionResult);
        const colorAnalysisResult = await imageProcessor.colorAnalysis.analyzePalette(
            visionResult.imagePropertiesAnnotation?.dominantColors?.colors || []
        );
        const qualityMetrics = await qualityAssessment.assessQuality(optimizedBuffer);

        const analysis = {
            tattooFeatures,
            imageQuality: qualityMetrics,
            colorAnalysis: colorAnalysisResult,
            visionAnalysis: {
                labels: visionResult.labelAnnotations || [],
                objects: visionResult.localizedObjectAnnotations || [],
                text: visionResult.textAnnotations || [],
                properties: visionResult.imagePropertiesAnnotation || {}
            },
            metadata: {
                ...file.metadata,
                path: file.path,
                hash: cacheKey,
                processedAt: new Date().toISOString(),
                fileStats: {
                    size: fs.statSync(file.path).size,
                    optimizedSize: optimizedBuffer.length
                },
                processingStages: {
                    optimization: true,
                    visionAnalysis: true,
                    featureAnalysis: true
                }
            }
        };

        await supabase
            .from('image_analysis')
            .upsert({
                path: file.path,
                analysis: analysis,
                analyzed_at: new Date().toISOString()
            });

        return analysis;

    } catch (error) {
        logger.error('Error in image processing:', {
            stage: processingStage,
            error: error.message,
            path: file?.path
        });

        await supabase
            .from('failed_analysis')
            .insert({
                path: file?.path,
                error: error.message,
                stage: processingStage,
                attempted_at: new Date().toISOString()
            });

        throw error;

    } finally {
        if (tempPath && fs.existsSync(tempPath)) {
            fs.unlinkSync(tempPath);
            logger.info('Cleaned up temp file:', { path: tempPath });
        }
        optimizedBuffer = null;
        if (global.gc) {
            global.gc();
        }
    }
}


async function retryAnalysis(operation, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            if (attempt === maxRetries - 1) throw error;
            await new Promise(resolve => 
                setTimeout(resolve, CACHE_SETTINGS.RETRY_DELAY_BASE * Math.pow(2, attempt))
            );
        }
    }
}

// Sequence Detection and Analysis
const sequenceDetector = {
    async findRelatedImages(imagePath, imageSignatures) {
logger.info('Starting sequence detection:', {
        targetPath: imagePath
    });       

 const currentImage = imageSignatures.get(imagePath);
if (!currentImage) {
        logger.info('No image signature found:', { path: imagePath });
        return [];
    }

        const directory = path.dirname(imagePath);
        const similarityThreshold = TATTOO_ANALYSIS_THRESHOLDS.SEQUENCE_MATCHING.VISUAL_SIMILARITY;

        // Find all images in same directory
        const dirImages = Array.from(imageSignatures.values())
            .filter(img => path.dirname(img.path) === directory);

 logger.info('Found directory images:', {
        directory,
        count: dirImages.length
    });

        // Sort by timestamp
        dirImages.sort((a, b) => 
            new Date(a.metadata.modified) - new Date(b.metadata.modified)
        );

        // Find position in sequence
        const sequencePosition = dirImages.findIndex(img => img.path === imagePath);

// Check for sequence patterns in filenames
    const sequencePattern = this.detectSequencePattern(dirImages.map(img => img.path));
    
    if (sequencePattern) {
        logger.info('Detected sequence pattern:', { pattern: sequencePattern });
    }

const sequence = dirImages.map((img, index) => ({
        ...img,
        sequenceInfo: {
            position: index + 1,
            total: dirImages.length,
            pattern: sequencePattern,
            timeDifference: index > 0 ? 
                new Date(img.metadata.modified) - new Date(dirImages[index - 1].metadata.modified) : 0
        }
    }));

    return sequence;
}, 

detectSequencePattern(paths) {
    // Common patterns in tattoo removal documentation
    const patterns = [
        /before|after/i,
        /t[0-9]+/i,  // T1, T2, etc.
        /session[0-9]+/i,
        /treatment[0-9]+/i,
        /\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b/ // dates
    ];

    for (const pattern of patterns) {
        if (paths.some(path => pattern.test(path))) {
            return pattern.toString();
        }
    }
    return null;
},



    calculateSequenceMetrics(sequence) {
        if (sequence.length < 2) return null;

        const first = sequence[0];
        const last = sequence[sequence.length - 1];

        return {
            duration: new Date(last.metadata.modified) - new Date(first.metadata.modified),
            totalSessions: sequence.length,
            overallProgress: this.calculateOverallProgress(sequence),
            stageProgression: this.analyzeStageProgression(sequence)
        };
    },

    calculateOverallProgress(sequence) {
        const progressMetrics = [];
        for (let i = 1; i < sequence.length; i++) {
            const current = sequence[i];
            const previous = sequence[i - 1];
            
            progressMetrics.push({
                timeDelta: new Date(current.metadata.modified) - new Date(previous.metadata.modified),
                changes: this.calculateImageChanges(previous, current)
            });
        }

        return {
            totalChange: progressMetrics.reduce((sum, metric) => sum + metric.changes.total, 0),
            averageChangeRate: progressMetrics.reduce((sum, metric) => 
                sum + (metric.changes.total / (metric.timeDelta / (1000 * 60 * 60 * 24))), 0) / progressMetrics.length
        };
    },

    calculateImageChanges(img1, img2) {
        const intensityChange = this.calculateIntensityChange(img1.imageStats, img2.imageStats);
        const patternChange = this.calculatePatternChange(img1.tattooFeatures, img2.tattooFeatures);
        
        return {
            intensity: intensityChange,
            pattern: patternChange,
            total: (intensityChange + patternChange) / 2
        };
    },

    calculateIntensityChange(stats1, stats2) {
        const intensity1 = stats1.means.reduce((a, b) => a + b, 0) / stats1.means.length;
        const intensity2 = stats2.means.reduce((a, b) => a + b, 0) / stats2.means.length;
        return Math.abs(intensity2 - intensity1) / 255;
    },

    calculatePatternChange(features1, features2) {
        if (!features1 || !features2) return 0;
        
        const patterns1 = features1.patterns || {};
        const patterns2 = features2.patterns || {};
        
        return Math.abs(
            (patterns2.dotPattern?.confidence || 0) - 
            (patterns1.dotPattern?.confidence || 0)
        );
    },

    analyzeStageProgression(sequence) {
        return sequence.map((image, index) => ({
            stage: image.tattooFeatures?.removal?.stage || 'unknown',
            timeFromStart: index === 0 ? 0 : 
                new Date(image.metadata.modified) - new Date(sequence[0].metadata.modified),
            metrics: image.tattooFeatures?.metrics || {}
        }));
    }
};

// Add these helper functions before initializeImageCache
async function storeSignature(path, signature) {
    try {
logger.debug('Signature data check:', {
            service: "tatt2awai-bot",
            path: signature.path,
            hasImageData: !!signature.imageData,
            signatureKeys: Object.keys(signature)
        });

        imageSignatures.set(path, signature);
        await supabase
            .from('image_signatures')
            .upsert({
                path,
                signature,
                analyzed_at: new Date().toISOString()
            });
    } catch (error) {
        logger.error('Error storing signature:', error);
        throw error;
    }
}

async function recordFailedAnalysis(path, error) {
    try {
        await supabase
            .from('failed_analysis')
            .insert({
                path,
                error: error.message,
                attempted_at: new Date().toISOString()
            });
    } catch (dbError) {
        logger.error('Error recording failed analysis:', {
            path,
            error: dbError.message
        });
    }
}

async function updateProgress(processed, failed, error = null) {
    try {
        await supabase
            .from('initialization_progress')
            .upsert({
                id: 'latest',
                processed_paths: Array.from(processed),
                failed_paths: Array.from(failed),
                last_error: error,
                updated_at: new Date().toISOString()
            });
    } catch (error) {
        logger.error('Error updating progress:', error);
    }
}

async function getDropboxAuth() {
    const currentTime = Date.now();
    
    // If we have a valid token that's not expired, reuse it
    if (dropboxAuthToken && lastAuthTime && 
        (currentTime - lastAuthTime) < AUTH_TOKEN_EXPIRY) {
        return dropboxAuthToken;
    }

    // Get new token
    try {
        const dropboxStatus = await dropboxManager.ensureAuth();
        if (!dropboxStatus) {
            throw new Error('Failed to authenticate with Dropbox');
        }
        
        dropboxAuthToken = dropboxStatus;
        lastAuthTime = currentTime;
        
        logger.info('Obtained new Dropbox authentication token');
        return dropboxAuthToken;
    } catch (error) {
        logger.error('Error authenticating with Dropbox:', error);
        throw error;
    }
}

// Also add this if not already present
function generateUUID(input) {
    return crypto.createHash('sha256').update(input).digest('hex').slice(0, 32);
}

function validateSignatureStructure(signature) {
    if (!signature || typeof signature !== 'object') return false;

    // Check required fields
    const requiredFields = ['metadata', 'analysis', 'aspectRatio'];
    const hasRequiredFields = requiredFields.some(field => {
        const value = signature[field];
        return value !== undefined && value !== null && 
               (typeof value === 'object' || typeof value === 'number');
    });

    if (!hasRequiredFields) return false;

    // Validate metadata if present
    if (signature.metadata) {
        const requiredMetadata = ['path', 'modified', 'analyzedAt'];
        const hasRequiredMetadata = requiredMetadata.some(field => 
            signature.metadata[field] !== undefined && 
            signature.metadata[field] !== null
        );

        if (!hasRequiredMetadata) return false;
    }

    // Additional validation for analysis data
    if (signature.analysis) {
        const hasValidAnalysis = (
            typeof signature.analysis === 'object' &&
            signature.analysis !== null &&
            (signature.analysis.tattooFeatures || 
             signature.analysis.imageQuality || 
             signature.analysis.colorAnalysis)
        );

        if (!hasValidAnalysis) return false;
    }

    return true;
}


           // Initialize processed signature

// Helper function to safely inspect objects
// Helper function for safe object inspection
function safeInspect(obj) {
    try {
        if (!obj || typeof obj !== 'object') return String(obj);
        
        const safeObj = {};
        for (const [key, value] of Object.entries(obj)) {
            if (Buffer.isBuffer(value)) {
                safeObj[key] = `Buffer(${value.length})`;
            } else if (value instanceof Uint8Array) {
                safeObj[key] = `Uint8Array(${value.length})`;
            } else if (typeof value === 'object') {
                safeObj[key] = safeInspect(value);
            } else {
                safeObj[key] = String(value);
            }
        }
        return safeObj;
    } catch (error) {
        return 'Error inspecting object';
    }
}

// Main signature processing function
async function processSignature(path, signature, searchBuffer, cache) {
    try {
        // Try to get image data (from cache or Dropbox)
        const candidateBuffer = await getImageData(path, signature, cache);
        if (!candidateBuffer) return null;

        // Compare images
        const visualSimilarity = await imageProcessor.enhancedCompareImages(
            searchBuffer,
            candidateBuffer,
            { includeFeatures: true }
        );

        return {
            path,
            signature,
            confidence: visualSimilarity.confidence || visualSimilarity.score,
            similarity: visualSimilarity.score,
            matchType: 'visual',
            details: visualSimilarity.details
        };
    } catch (error) {
        logger.warn('Error processing signature:', {
            service: "tatt2awai-bot",
            path,
            error: error.message
        });
        return null;
    }
}

async function getImageData(path, signature, cache) {
    try {
        // Check cache first
        if (cache.has(path)) {
            return cache.get(path);
        }

        // Try to get from Dropbox with retries
        let attempts = 0;
        const maxAttempts = 3;
        
        while (attempts < maxAttempts) {
            try {
                const fileData = await dropboxManager.downloadFile(path);
                
                if (fileData?.result?.fileBinary) {
                    const buffer = Buffer.isBuffer(fileData.result.fileBinary) ? 
                        fileData.result.fileBinary :
                        Buffer.from(fileData.result.fileBinary);
                    
                    cache.set(path, buffer);
                    return buffer;
                }
                throw new Error('No binary data received');
                
            } catch (error) {
                attempts++;
                
                if (error.status === 401) {
                    await dropboxManager.refreshAccessToken();
                    continue;
                }
                
                if (attempts === maxAttempts) {
                    throw error;
                }
                
                await new Promise(resolve => 
                    setTimeout(resolve, 1000 * Math.pow(2, attempts))
                );
            }
        }
        
    } catch (error) {
        logger.error('Failed to get image data:', {
            path,
            error: error.message,
            stack: error.stack
        });
        return null;
    }
}

async function retryOperation(operation, maxRetries = 3, baseDelay = 2000) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            if (attempt === maxRetries) {
                logger.error('Operation failed after max retries:', {
                    service: "tatt2awai-bot",
                    error: error.message,
                    maxRetries
                });
                throw error;
            }
            
            const delay = baseDelay * Math.pow(2, attempt - 1);
            logger.info('Retrying operation:', {
                service: "tatt2awai-bot",
                attempt,
                nextDelay: delay,
                error: error.message
            });
            
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

async function downloadFromDropbox(path) {
    const fileData = await dropboxManager.downloadFile(path);
    if (!fileData?.result?.fileBinary) {
        throw new Error('No binary data received from Dropbox');
    }
    return Buffer.isBuffer(fileData.result.fileBinary) ?
        fileData.result.fileBinary :
        Buffer.from(fileData.result.fileBinary);
}

async function updateSignatureInBackground(path, signature, buffer, entryMetadata) {
    try {
        const image = sharp(buffer);
        const [stats, metadata] = await Promise.all([
            image.stats(),
            image.metadata()
        ]);

        const updatedSignature = {
            path,
            imageData: buffer,
            aspectRatio: metadata.width / metadata.height,
            means: stats.channels.map(c => c.mean),
            stdDevs: stats.channels.map(c => c.std),
            metadata: {
                modified: entryMetadata.server_modified,
                size: entryMetadata.size,
                analyzedAt: new Date().toISOString(),
                width: metadata.width,
                height: metadata.height,
                format: metadata.format
            }
        };

        // Update in Supabase without waiting
        supabase
            .from('image_signatures')
            .upsert({
                path,
                signature: {
                    ...updatedSignature,
                    imageData: null  // Don't store binary data
                },
                analyzed_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                status: 'processed'
            })
            .then(({ error }) => {
                if (error) {
                    logger.warn(`Supabase update failed for ${path}:`, {
                        service: "tatt2awai-bot",
                        error: error.message
                    });
                }
            });

    } catch (error) {
        logger.error(`Error updating signature for ${path}:`, {
            service: "tatt2awai-bot",
            error: error.message
        });
    }
}


// Signature structure validation function
function validateSignatureStructure(signature) {
    if (!signature || typeof signature !== 'object') return false;

    const requiredFields = ['metadata', 'analysis', 'aspectRatio'];
    const hasRequiredFields = requiredFields.some(field => {
        const value = signature[field];
        return value !== undefined && value !== null && 
               (typeof value === 'object' || typeof value === 'number');
    });

    if (!hasRequiredFields) return false;

    // Validate metadata if present
    if (signature.metadata) {
        const requiredMetadata = ['path', 'modified', 'analyzedAt'];
        const hasRequiredMetadata = requiredMetadata.some(field => 
            signature.metadata[field] !== undefined && 
            signature.metadata[field] !== null
        );

        if (!hasRequiredMetadata) return false;
    }

    return true;
}

async function validateAndProcessImage(tempPath, fileData, entry) {
    try {

 const existingSignature = await supabase
            .from('image_signatures')
            .select('signature')
            .eq('path', entry.path_lower)
            .single();

if (existingSignature?.data?.signature) {
    try {
        // Get the image data even for cached signatures
        const fileData = await dropboxManager.downloadFile(entry.path_lower);
        if (fileData?.result?.fileBinary) {
            const imageBuffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                fileData.result.fileBinary :
                Buffer.from(fileData.result.fileBinary);

            // Add image data to the cached signature
            const signatureWithImage = {
                ...existingSignature.data.signature,
                imageData: imageBuffer
            };

            imageSignatures.set(entry.path_lower, signatureWithImage);
            processedEntries.add(entry.path_lower);
            
            logger.info(`Added image data to cached signature for ${entry.path_lower}`, {
                service: "tatt2awai-bot",
                hasImageData: !!imageBuffer
            });
            
            return {
                path: entry.path_lower,
                status: 'cached_with_image',
                timestamp: new Date().toISOString()
            };
        }
 } catch (error) {
        logger.warn(`Failed to add image data to cached signature ${entry.path_lower}:`, {
            service: "tatt2awai-bot",
            error: error.message
        });
    }
}

        // First try to validate the image
        const image = sharp(tempPath);
        await image.metadata();  // This will throw if image is corrupted

        // If validation passes, proceed with processing
        const [stats, metadata] = await Promise.all([
            image.stats().catch(e => {
                logger.warn(`Stats calculation failed for ${entry.path_lower}`, {
                    service: "tatt2awai-bot",
                    error: e.message,
                    type: 'stats_error'
                });
                return null;
            }),
            image.metadata().catch(e => {
                logger.warn(`Metadata extraction failed for ${entry.path_lower}`, {
                    service: "tatt2awai-bot",
                    error: e.message,
                    type: 'metadata_error'
                });
                return null;
            })
        ]);

        // If either stats or metadata failed, throw error
        if (!stats || !metadata) {
            throw new Error('Failed to extract image information');
        }

        // Generate a deterministic signature ID
        const signatureId = crypto.createHash('sha256')
            .update(`${entry.path_lower}-${metadata.width}-${metadata.height}-${stats.channels[0].mean}`)
            .digest('hex');

        // Create the signature object
// Generate perceptual hashes for search
const perceptualHashes = await Promise.all([16, 32, 64].map(size =>
    new Promise((resolve, reject) => {
        imageHash(tempPath, size, true, (error, hash) => {
            if (error) reject(error);
            else resolve({ size, hash });
        });
    })
));

// Create the signature object
const signature = {
    signatureId,
    path: entry.path_lower,
    imageData: fileData.result.fileBinary,
    aspectRatio: metadata.width / metadata.height,
    means: stats.channels.map(c => c.mean),
    stdDevs: stats.channels.map(c => c.std),
    perceptualHashes, // Add the hashes
    colorMetrics: {
        channelStats: stats.channels.map(c => ({
            mean: c.mean,
            std: c.std,
            min: c.min,
            max: c.max
        })),
        dominantColors: await extractDominantColors(image)
    },

            metadata: {
                modified: entry.server_modified,
                size: entry.size,
                analyzedAt: new Date().toISOString(),
                width: metadata.width,
                height: metadata.height,
                format: metadata.format,
                processingDetails: {
                    version: '2.0',
                    processingTime: Date.now(),
                    features: ['basic', 'color', 'metrics']
                }
            }
        };

imageSignatures.set(entry.path_lower, signature);

// Store in Supabase without the image data
await supabase
    .from('image_signatures')
    .upsert({
        path: entry.path_lower,
        signature: {
            ...signature,
            imageData: null  // Don't store binary data in Supabase
        },
        analyzed_at: new Date().toISOString()
    });

        // Log signature details
        logger.info('Generated signature details:', {
            service: "tatt2awai-bot",
            path: entry.path_lower,
            signatureId,
            dimensions: {
                width: metadata.width,
                height: metadata.height,
                aspectRatio: metadata.width / metadata.height
            },
            features: {
                hasColorMetrics: true,
                hasStats: true,
                format: metadata.format
            },
            timestamp: new Date().toISOString()
        });

        return signature;

    } catch (error) {
        if (error.message.includes('Corrupt JPEG') ||
            error.message.includes('Input buffer contains unsupported image format')) {
            // Attempt to repair the image
            try {
                logger.info(`Attempting to repair corrupted image ${entry.path_lower}`, {
                    service: "tatt2awai-bot"
                });
                
                const repairedBuffer = await sharp(tempPath)
                    .jpeg({ quality: 90 })
                    .toBuffer();
                
                const repairedImage = sharp(repairedBuffer);
                const [repairedStats, repairedMetadata] = await Promise.all([
                    repairedImage.stats(),
                    repairedImage.metadata()
                ]);

                // Generate signature ID for repaired image
                const repairedSignatureId = crypto.createHash('sha256')
                    .update(`${entry.path_lower}-repaired-${repairedMetadata.width}-${repairedMetadata.height}`)
                    .digest('hex');

                const repairedSignature = {
                    signatureId: repairedSignatureId,
                    path: entry.path_lower,
                    imageData: repairedBuffer,
                    aspectRatio: repairedMetadata.width / repairedMetadata.height,
                    means: repairedStats.channels.map(c => c.mean),
                    stdDevs: repairedStats.channels.map(c => c.std),
                    colorMetrics: {
                        channelStats: repairedStats.channels.map(c => ({
                            mean: c.mean,
                            std: c.std,
                            min: c.min,
                            max: c.max
                        })),
                        dominantColors: await extractDominantColors(repairedImage)
                    },
                    metadata: {
                        modified: entry.server_modified,
                        size: repairedBuffer.length,
                        analyzedAt: new Date().toISOString(),
                        width: repairedMetadata.width,
                        height: repairedMetadata.height,
                        format: 'jpeg',
                        wasRepaired: true,
                        processingDetails: {
                            version: '2.0',
                            processingTime: Date.now(),
                            features: ['basic', 'color', 'metrics'],
                            repairApplied: true
                        }
                    }
                };

                logger.info('Generated repaired signature:', {
                    service: "tatt2awai-bot",
                    path: entry.path_lower,
                    signatureId: repairedSignatureId,
                    wasRepaired: true,
                    timestamp: new Date().toISOString()
                });

                return repairedSignature;

            } catch (repairError) {
                logger.error(`Failed to repair corrupted image ${entry.path_lower}:`, {
                    service: "tatt2awai-bot",
                    error: repairError.message,
                    type: 'repair_failed'
                });
                throw new Error(`Image corruption could not be repaired: ${error.message}`);
            }
        }
        throw error;
    }
}

// Helper function to extract dominant colors
async function extractDominantColors(image) {
    try {
        const { dominant } = await image.stats();
        return {
            background: dominant.background,
            foreground: dominant.foreground,
            mean: dominant.mean
        };
    } catch (error) {
        logger.warn('Failed to extract dominant colors:', {
            service: "tatt2awai-bot",
            error: error.message
        });
        return null;
    }
}

// Cache Initialization and Management
async function initializeImageCache() {
    const startTime = Date.now();
    const processedEntries = new Set();
    const failedEntries = new Set();
    const BATCH_SIZE = 50;
    const CONCURRENT_LIMIT = 5;
    const RETRY_LIMIT = 3;
    const RETRY_DELAY = 2000;
    let currentBatch = 0;
    let totalBatches = 0;

    try {
        logger.info('Starting cache initialization check...', {
            service: "tatt2awai-bot",
            timestamp: new Date().toISOString()
        });

        // Check existing cache in Supabase first
        const { count, error: cacheError } = await supabase
            .from('image_signatures')
            .select('*', { count: 'exact', head: true });

        logger.info('Cache status check:', {
            service: "tatt2awai-bot",
            hasCache: !cacheError && count > 0,
            existingCount: count || 0,
            timestamp: new Date().toISOString()
        });

        // If we have cached entries, load them
        if (!cacheError && count > 0) {
            logger.info('Loading existing cache...', {
                service: "tatt2awai-bot",
                existingEntries: count
            });

            let page = 0;
            const pageSize = 1000;

            while (true) {
                const { data: signatures, error } = await supabase
                    .from('image_signatures')
                    .select('*')
                    .range(page * pageSize, (page + 1) * pageSize - 1);

                if (error || !signatures || signatures.length === 0) break;

                signatures.forEach(sigRecord => {
                    if (sigRecord && sigRecord.path && sigRecord.signature) {
                        imageSignatures.set(sigRecord.path, sigRecord.signature);
                        processedEntries.add(sigRecord.path);
                    }
                });

                logger.info('Loaded signature chunk:', {
                    service: "tatt2awai-bot",
                    chunk: page + 1,
                    chunkSize: signatures.length,
                    totalLoaded: imageSignatures.size,
                    progress: `${Math.round((imageSignatures.size / count) * 100)}%`
                });

                if (signatures.length < pageSize) {
                    logger.info('Reached end of signatures', {
                        service: "tatt2awai-bot"
                    });
                    break;
                }
                page++;
            }
        }

        // Now fetch from Dropbox and update/add missing entries
        const entries = await dropboxManager.fetchDropboxEntries('');
        if (!entries?.result?.entries) {
            throw new Error('No entries returned from Dropbox');
        }

        const imageFiles = entries.result.entries.filter(entry =>
            CACHE_SETTINGS.SUPPORTED_FORMATS.some(ext =>
                entry.path_lower.endsWith(ext)
            )
        );

        totalBatches = Math.ceil(imageFiles.length / BATCH_SIZE);

        for (let i = 0; i < imageFiles.length; i += BATCH_SIZE) {
            currentBatch = Math.floor(i / BATCH_SIZE) + 1;
            const batch = imageFiles.slice(i, Math.min(i + BATCH_SIZE, imageFiles.length));

            for (let j = 0; j < batch.length; j += CONCURRENT_LIMIT) {
                const currentChunk = batch.slice(j, j + CONCURRENT_LIMIT);

                await Promise.all(currentChunk.map(async (entry) => {
                    try {
                        if (imageSignatures.has(entry.path_lower) &&
                            imageSignatures.get(entry.path_lower).metadata?.analyzedAt) {
                            return {
                                path: entry.path_lower,
                                status: 'skipped',
                                timestamp: new Date().toISOString()
                            };
                        }

                        const tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(entry.path_lower)}`);

                        try {
                            const fileData = await dropboxManager.downloadFile(entry.path_lower);
                            if (!fileData?.result?.fileBinary) {
                                throw new Error('Invalid file data from Dropbox');
                            }

                            fs.writeFileSync(tempPath, fileData.result.fileBinary);

                            const newSignature = {
                                path: entry.path_lower,
                                imageData: fileData.result.fileBinary,
                                metadata: {
                                    path: entry.path_lower,
                                    modified: entry.server_modified,
                                    size: entry.size,
                                    analyzedAt: new Date().toISOString()
                                }
                            };

                            imageSignatures.set(entry.path_lower, newSignature);
                            processedEntries.add(entry.path_lower);

                            await supabase
                                .from('image_signatures')
                                .upsert({
                                    path: entry.path_lower,
                                    signature: {
                                        ...newSignature,
                                        imageData: null
                                    },
                                    analyzed_at: new Date().toISOString()
                                });

                            return {
                                path: entry.path_lower,
                                status: 'success',
                                timestamp: new Date().toISOString()
                            };

                        } catch (error) {
                            logger.error(`Error processing ${entry.path_lower}:`, {
                                service: "tatt2awai-bot",
                                error: error.message,
                                stack: error.stack
                            });
                            failedEntries.add(entry.path_lower);
                            return {
                                path: entry.path_lower,
                                status: 'failed',
                                error: error.message
                            };
                        } finally {
                            if (fs.existsSync(tempPath)) {
                                fs.unlinkSync(tempPath);
                            }
                        }
                    } catch (error) {
                        logger.error(`Error processing entry ${entry.path_lower}:`, {
                            service: "tatt2awai-bot",
                            error: error.message,
                            stack: error.stack
                        });
                        return null;
                    }
                }));
            }

            const progress = ((currentBatch / totalBatches) * 100).toFixed(1);
            const timeElapsed = Date.now() - startTime;
            const estimatedTimeRemaining = Math.round(
                (timeElapsed / currentBatch) * (totalBatches - currentBatch) / 1000
            );

            logger.info('Cache initialization progress:', {
                service: "tatt2awai-bot",
                progress: `${progress}%`,
                currentBatch,
                totalBatches,
                processed: processedEntries.size,
                failed: failedEntries.size,
                timeElapsed: `${Math.round(timeElapsed / 1000)}s`,
                estimatedTimeRemaining: `${estimatedTimeRemaining}s`,
                memoryUsage: Math.round(process.memoryUsage().heapUsed / (1024 * 1024))
            });
        }

        isInitialized = true;
        return true;

    } catch (error) {
        logger.error('Cache initialization failed:', {
            service: "tatt2awai-bot",
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

async function loadChunkWithRetry(offset, pageSize, maxRetries, retryDelay) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const { data, error } = await supabase
                .from('image_signatures')
                .select('*')
                .range(offset, offset + pageSize - 1);

            if (error) throw error;
            return data;
        } catch (error) {
            if (attempt === maxRetries - 1) throw error;
            logger.warn(`Retrying chunk load (attempt ${attempt + 1}/${maxRetries}):`, {
                error: error.message,
                offset,
                pageSize
            });
            await new Promise(resolve => setTimeout(resolve, retryDelay * Math.pow(2, attempt)));
        }
    }
}
            // Load all signatures first
// Replace the while(true) loop section with this updated version

// If we reach here, no existing cache was found

// Health Check Endpoints
app.get('/cache/status', async (req, res) => {
    const { data: count } = await supabase
        .from('image_signatures')
        .select('count', { count: 'exact' });
        
    res.json({
        total: count,
        isInitialized,
        lastUpdate: new Date().toISOString()
    });
});

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
        res.status(500).json({ error: error.message });
    }
});

app.get('/debug/signature/:path', async (req, res) => {
    const signature = imageSignatures.get(req.params.path);
    res.json({
        hasSignature: !!signature,
        details: signature ? {
            aspectRatio: signature.aspectRatio,
            hasMeans: !!signature.means,
            hasFeatures: !!signature.features,
            path: signature.path
        } : null
    });
});

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
        res.status(500).json({ error: error.message });
    }
});

// Image Upload and Processing Endpoints
app.post('/upload', upload.single('image'), async (req, res) => {
    const processingId = crypto.randomUUID();
    console.log('1. Upload request start:', { processingId });
    
    if (!req.file) {
        return res.status(400).json({ error: 'No file provided' });
    }

    try {
        const dropboxStatus = await dropboxManager.ensureAuth();
        const [visionResult] = await visionClient.labelDetection(req.file.path);
        
        // Store in documents table
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

        // Process image
        const analysis = await processImage(req.file);
        
        res.json({
            success: true,
            processingId,
            analysis,
            document: document.id
        });

    } catch (error) {
        logger.error('Processing error:', error);
        res.status(500).json({ error: error.message });
    } finally {
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
    }
});


async function generateColorProfile(image) {
    try {
        const stats = await image.stats();
        // stats.channels contains the color information
        return stats.channels.map(channel => ({
            rgb: {
                red: channel.mean,
                green: channel.mean,
                blue: channel.mean
            },
            prominence: channel.mean / 255
        }));
    } catch (error) {
        logger.error('Error generating color profile:', {
            error: error.message,
            stack: error.stack
        });
        return [];
    }
}

// Enhanced signature comparison with detailed logging

function compareFeatures(features1, features2) {
    let featureScore = 0;
    
    if (features1.inkPatterns && features2.inkPatterns) {
        const densityDiff = Math.abs(features1.inkPatterns.inkDensity - features2.inkPatterns.inkDensity);
        featureScore += 0.5 * (1 - densityDiff);
    }
    
    return featureScore;
}

// Add this before the visual search endpoint to inspect the cache
app.get('/debug/cache', (req, res) => {
    const cacheInfo = {
        size: imageSignatures.size,
        sample: Array.from(imageSignatures.entries())
            .slice(0, 5)
            .map(([path, sig]) => ({
                path,
                hasAspectRatio: !!sig.aspectRatio,
                hasMeans: !!sig.means,
                hasStdDevs: !!sig.stdDevs,
                aspectRatio: sig.aspectRatio
            }))
    };
    res.json(cacheInfo);
});


function compareColorProfiles(profile1, profile2) {
    try {
        if (!profile1 || !profile2) return 0;
        
        // Compare dominant colors
        const colors1 = profile1.sort((a, b) => b.prominence - a.prominence);
        const colors2 = profile2.sort((a, b) => b.prominence - a.prominence);
        
        let colorScore = 0;
        const minColors = Math.min(colors1.length, colors2.length);
        
        for (let i = 0; i < minColors; i++) {
            const color1 = colors1[i];
            const color2 = colors2[i];
            
            const colorDiff = Math.abs(color1.rgb.red - color2.rgb.red) +
                             Math.abs(color1.rgb.green - color2.rgb.green) +
                             Math.abs(color1.rgb.blue - color2.rgb.blue);
            
            colorScore += (1 - colorDiff / (3 * 255)) * color1.prominence;
        }
        
        return colorScore / minColors;
    } catch (error) {
        logger.error('Error comparing color profiles:', error);
        return 0;
    }
}

async function getFinalVerificationScore(searchImage, candidateImage) {
    try {
        // Convert both images to same size for direct comparison
        const normalizedSearch = await sharp(searchImage)
            .resize(300, 300, { fit: 'contain' })
            .greyscale()
            .normalize()
            .raw()
            .toBuffer();
        
        const normalizedCandidate = await sharp(candidateImage)
            .resize(300, 300, { fit: 'contain' })
            .greyscale()
            .normalize()
            .raw()
            .toBuffer();

        // Compare structural similarity
        let diffScore = 0;
        for (let i = 0; i < normalizedSearch.length; i++) {
            diffScore += Math.abs(normalizedSearch[i] - normalizedCandidate[i]);
        }
        
        return 1 - (diffScore / (normalizedSearch.length * 255));
    } catch (error) {
        logger.error('Final verification error:', error);
        return 0;
    }
}

function compareSignatures(sig1, sig2) {
    try {
        let score = 0;
        const weights = {
            aspectRatio: 0.05,
            means: 0.10,
            tattooFeatures: 0.35,
            placement: 0.15,
            inkColors: 0.15,
            geometryMatch: 0.20
        };

        // Structural pattern comparison
        if (sig1.analysis?.tattooFeatures?.patterns && sig2.analysis?.tattooFeatures?.patterns) {
            const pattern1 = sig1.analysis.tattooFeatures.patterns;
            const pattern2 = sig2.analysis.tattooFeatures.patterns;
            
            // Compare pattern types and characteristics
            const patternTypes1 = new Set(pattern1.types || []);
            const patternTypes2 = new Set(pattern2.types || []);
            const commonPatterns = new Set([...patternTypes1].filter(x => patternTypes2.has(x)));
            
            const patternScore = commonPatterns.size / Math.max(patternTypes1.size, patternTypes2.size);
            score += weights.geometryMatch * patternScore;

            // Compare density distributions
            if (pattern1.densityMap && pattern2.densityMap) {
                const densityDiff = pattern1.densityMap.map((val, idx) => 
                    Math.abs(val - (pattern2.densityMap[idx] || 0))
                ).reduce((a, b) => a + b, 0) / pattern1.densityMap.length;
                
                score += weights.geometryMatch * (1 - densityDiff);
            }
        }

        // Enhanced color analysis
        if (sig1.analysis?.colorAnalysis && sig2.analysis?.colorAnalysis) {
            const colorScore = compareColorDistributions(
                sig1.analysis.colorAnalysis,
                sig2.analysis.colorAnalysis
            );
            score += weights.inkColors * colorScore;
        }

        // Compare dominant features
        if (sig1.analysis?.dominantFeatures && sig2.analysis?.dominantFeatures) {
            const featureMatch = compareDominantFeatures(
                sig1.analysis.dominantFeatures,
                sig2.analysis.dominantFeatures
            );
            score += weights.tattooFeatures * featureMatch;
        }

        // Location and placement verification
        if (sig1.analysis?.placement && sig2.analysis?.placement) {
            const placementScore = comparePlacement(
                sig1.analysis.placement,
                sig2.analysis.placement
            );
            score += weights.placement * placementScore;
        }

        return {
            similarity: score,
            filePath: sig2.path,
            fileName: sig2.path ? path.basename(sig2.path) : null,
            matchDetails: {
                patternMatch: patternScore || 0,
                colorMatch: colorScore || 0,
                featureMatch: featureMatch || 0,
                placementMatch: placementScore || 0
            }
        };
    } catch (error) {
        logger.error('Signature comparison error:', error);
        return {
            similarity: 0,
            filePath: null,
            fileName: null
        };
    }
}

// Helper functions for detailed comparison
function compareColorDistributions(colors1, colors2) {
    const getColorFingerprint = (colors) => {
        return colors.reduce((fp, color) => ({
            red: fp.red + (color.rgb?.red || 0) * color.prominence,
            green: fp.green + (color.rgb?.green || 0) * color.prominence,
            blue: fp.blue + (color.rgb?.blue || 0) * color.prominence,
            variance: fp.variance + color.prominence
        }), { red: 0, green: 0, blue: 0, variance: 0 });
    };

    const fp1 = getColorFingerprint(colors1);
    const fp2 = getColorFingerprint(colors2);

    const colorDiff = Math.sqrt(
        Math.pow(fp1.red - fp2.red, 2) +
        Math.pow(fp1.green - fp2.green, 2) +
        Math.pow(fp1.blue - fp2.blue, 2)
    ) / 441.67;

    const varianceDiff = Math.abs(fp1.variance - fp2.variance);
    
    return (1 - colorDiff) * 0.7 + (1 - varianceDiff) * 0.3;
}

function compareDominantFeatures(features1, features2) {
    const compareFeatures = (f1, f2) => {
        const typeMatch = f1.type === f2.type ? 1 : 0;
        const sizeDiff = Math.abs(f1.size - f2.size);
        const positionDiff = Math.sqrt(
            Math.pow(f1.position.x - f2.position.x, 2) +
            Math.pow(f1.position.y - f2.position.y, 2)
        );
        
        return (typeMatch + (1 - sizeDiff) + (1 - positionDiff)) / 3;
    };

    const pairs = features1.map(f1 => {
        const bestMatch = features2
            .map(f2 => ({ feature: f2, score: compareFeatures(f1, f2) }))
            .reduce((best, curr) => curr.score > best.score ? curr : best, { score: 0 });
        return bestMatch.score;
    });

    return pairs.reduce((sum, score) => sum + score, 0) / pairs.length;
}

function comparePlacement(placement1, placement2) {
    const normalizedPlacement1 = placement1.toLowerCase().trim();
    const normalizedPlacement2 = placement2.toLowerCase().trim();
    
    // Direct match
    if (normalizedPlacement1 === normalizedPlacement2) return 1;
    
    // Related placement groups
    const placementGroups = {
        arm: ['upperarm', 'forearm', 'sleeve', 'shoulder'],
        leg: ['thigh', 'calf', 'ankle'],
        torso: ['chest', 'back', 'stomach', 'ribs']
    };
    
    for (const [group, locations] of Object.entries(placementGroups)) {
        const inGroup1 = locations.some(loc => normalizedPlacement1.includes(loc));
        const inGroup2 = locations.some(loc => normalizedPlacement2.includes(loc));
        if (inGroup1 && inGroup2) return 0.8;
    }
    
    return 0;
}

// Add these GLCM analysis functions
function calculateGLCMContrast(glcm) {
    let contrast = 0;
    for (let i = 0; i < glcm.length; i++) {
        for (let j = 0; j < glcm[i].length; j++) {
            contrast += Math.pow(i - j, 2) * glcm[i][j];
        }
    }
    return contrast;
}

function calculateGLCMCorrelation(glcm) {
    const size = glcm.length;
    let meanI = 0, meanJ = 0;
    let stdI = 0, stdJ = 0;
    
    // Calculate means
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            meanI += i * glcm[i][j];
            meanJ += j * glcm[i][j];
        }
    }
    
    // Calculate standard deviations
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            stdI += Math.pow(i - meanI, 2) * glcm[i][j];
            stdJ += Math.pow(j - meanJ, 2) * glcm[i][j];
        }
    }
    stdI = Math.sqrt(stdI);
    stdJ = Math.sqrt(stdJ);
    
    // Calculate correlation
    let correlation = 0;
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            correlation += ((i - meanI) * (j - meanJ) * glcm[i][j]) / (stdI * stdJ);
        }
    }
    return correlation;
}

function calculateGLCMEnergy(glcm) {
    let energy = 0;
    for (let i = 0; i < glcm.length; i++) {
        for (let j = 0; j < glcm[i].length; j++) {
            energy += Math.pow(glcm[i][j], 2);
        }
    }
    return energy;
}

function calculateGLCMHomogeneity(glcm) {
    let homogeneity = 0;
    for (let i = 0; i < glcm.length; i++) {
        for (let j = 0; j < glcm[i].length; j++) {
            homogeneity += glcm[i][j] / (1 + Math.abs(i - j));
        }
    }
    return homogeneity;
}

function calculateEdgeOrientation(data, width, height) {
    const gradients = calculateGradients(data, width, height);
    const orientations = calculateOrientations(gradients);
    return analyzeOrientations(orientations);
}

function calculateGradients(data, width, height) {
    const gradX = new Float32Array((width - 1) * (height - 1));
    const gradY = new Float32Array((width - 1) * (height - 1));

    for (let y = 0; y < height - 1; y++) {
        for (let x = 0; x < width - 1; x++) {
            const idx = y * width + x;
            gradX[y * (width - 1) + x] = data[idx + 1] - data[idx];
            gradY[y * (width - 1) + x] = data[idx + width] - data[idx];
        }
    }

    return { gradX, gradY };
}

// Orientation analysis
function calculateOrientations(gradients) {
    if (!gradients || !gradients.gradX || !gradients.gradY) {
        return new Float32Array(0);
    }
    
    const orientations = new Float32Array(gradients.gradX.length);
    
    for (let i = 0; i < gradients.gradX.length; i++) {
        orientations[i] = Math.atan2(gradients.gradY[i], gradients.gradX[i]);
    }
    
    return orientations;
}

function analyzeOrientations(orientations) {
    if (!orientations || orientations.length === 0) {
        return {
            dominant: 'unknown',
            distribution: new Array(8).fill(0),
            strength: 0
        };
    }

    const bins = new Array(8).fill(0);
    const binSize = Math.PI / 4;

    // Count orientations in 45-degree bins
    for (let angle of orientations) {
        // Normalize angle to [0, 2π)
        angle = angle < 0 ? angle + 2 * Math.PI : angle;
        const binIndex = Math.floor(angle / binSize) % 8;
        bins[binIndex]++;
    }

    // Find dominant orientation
    const maxBin = bins.indexOf(Math.max(...bins));
    const orientationMap = [
        'horizontal',
        'diagonal-up',
        'vertical',
        'diagonal-down',
        'horizontal',
        'diagonal-up',
        'vertical',
        'diagonal-down'
    ];

    return {
        dominant: orientationMap[maxBin],
        distribution: bins,
        strength: Math.max(...bins) / orientations.length
    };
}


function calculateNormalizedEntropy(normalizedStats) {
    // Create histogram using normal distribution approximation
    const bins = 256;
    const histogram = new Float32Array(bins);
    
    // Generate normalized histogram using Gaussian distribution
    for (let i = 0; i < bins; i++) {
        const x = i / (bins - 1); // Normalize to [0, 1]
        const z = (x - normalizedStats.mean) / (normalizedStats.stdev || 0.0001);
        histogram[i] = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * (normalizedStats.stdev || 0.0001));
    }

    // Normalize histogram to sum to 1
    const sum = histogram.reduce((a, b) => a + b, 0);
    for (let i = 0; i < bins; i++) {
        histogram[i] /= sum;
    }

    // Calculate entropy
    return -histogram.reduce((entropy, p) => 
        entropy + (p > 0 ? p * Math.log2(p) : 0), 0);
}


async function calculateImageStatistics(buffer) {
    try {
        const image = sharp(buffer);
        const [stats, metadata] = await Promise.all([
            image.stats(),
            image.metadata()
        ]);

        // Process channel statistics without using channel.values
        const channelStats = stats.channels.map(channel => {
            const range = channel.max - channel.min;
            const normalized = {
                mean: channel.mean / 255,
                stdev: channel.std / 255,
                min: channel.min / 255,
                max: channel.max / 255
            };

            return {
                mean: channel.mean,
                stdev: channel.std,
                min: channel.min,
                max: channel.max,
                entropy: calculateEntropy(normalized),
                range: range,
                variance: channel.std * channel.std
            };
        });

        return {
            channels: channelStats,
            aspectRatio: metadata.width / metadata.height,
            dimensions: {
                width: metadata.width,
                height: metadata.height
            },
            format: metadata.format,
            space: metadata.space,
            density: metadata.density,
            hasAlpha: metadata.hasAlpha,
            isProgressive: metadata.isProgressive,
            totalPixels: metadata.width * metadata.height
        };
    } catch (error) {
        logger.error('Error calculating image statistics:', {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

// Update the entropy calculation
function calculateEntropy(channel) {
    if (!channel || typeof channel.mean === 'undefined' || typeof channel.std === 'undefined') {
        return 0;
    }

    // Create histogram using normal distribution approximation
    const bins = 256;
    const histogram = new Float32Array(bins);
    const mean = channel.mean;
    const std = channel.std || 1; // Prevent division by zero
    
    // Generate normalized histogram using Gaussian distribution
    for (let i = 0; i < bins; i++) {
        const x = i / (bins - 1); // Normalize to [0, 1]
        const z = (x - mean / 255) / (std / 255);
        histogram[i] = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * (std / 255));
    }
    
    // Normalize histogram to sum to 1
    const sum = histogram.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;
    
    for (let i = 0; i < bins; i++) {
        histogram[i] /= sum;
    }
    
    // Calculate entropy
    let entropy = 0;
    for (let i = 0; i < bins; i++) {
        if (histogram[i] > 0) {
            entropy -= histogram[i] * Math.log2(histogram[i]);
        }
    }
    
    return entropy;
}


// Add ImageComparison.analyzeImageContent
ImageComparison.analyzeImageContent = async function(buffer) {
    try {
        const image = sharp(buffer);
        const { data, info } = await image
            .raw()
            .toBuffer({ resolveWithObject: true });

        // Basic content analysis
        const contentAnalysis = {
            brightness: calculateAverageBrightness(data),
            contrast: calculateImageContrast(data),
            edges: await detectEdges(buffer),
            patterns: detectPatterns(data, info.width, info.height)
        };

        return {
            ...contentAnalysis,
            type: determineContentType(contentAnalysis),
            confidence: calculateAnalysisConfidence(contentAnalysis)
        };
    } catch (error) {
        logger.error('Error in image content analysis:', error);
        return {
            error: true,
            message: error.message
        };
    }
};

function calculateAverageBrightness(data) {
    return data.reduce((sum, value) => sum + value, 0) / data.length;
}

function calculateImageContrast(data) {
    const brightness = calculateAverageBrightness(data);
    const variance = data.reduce((sum, val) => 
        sum + Math.pow(val - brightness, 2), 0) / data.length;
    return Math.sqrt(variance);
}

async function detectEdges(buffer) {
    const edgeImage = await sharp(buffer)
        .greyscale()
        .convolve({
            width: 3,
            height: 3,
            kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
        })
        .raw()
        .toBuffer();

    return {
        strength: calculateAverageBrightness(edgeImage),
        distribution: analyzeEdgeDistribution(edgeImage)
    };
}

function detectPatterns(data, width, height) {
    // Simple pattern detection
    return {
        horizontal: detectLinearPatterns(data, width, height, 'horizontal'),
        vertical: detectLinearPatterns(data, width, height, 'vertical'),
        diagonal: detectLinearPatterns(data, width, height, 'diagonal')
    };
}

function detectLinearPatterns(data, width, height, direction) {
    // Placeholder for pattern detection logic
    return {
        confidence: 0.8,
        count: 0
    };
}

function analyzeEdgeDistribution(edgeData) {
    // Placeholder for edge distribution analysis
    return {
        uniform: true,
        density: 0.5
    };
}

function determineContentType(analysis) {
    // Basic content type determination
    if (analysis.edges.strength > 100) {
        return 'high_detail';
    } else if (analysis.contrast > 50) {
        return 'medium_detail';
    }
    return 'low_detail';
}

function calculateAnalysisConfidence(analysis) {
    // Simple confidence calculation
    return 0.8;
}


// Helper functions first
const extractSequenceNumber = (filename) => {
    const patterns = [
        /t(\d+)/i,                    // t1, t2, etc.
        /session[_-]?(\d+)/i,         // session1, session_2
        /treatment[_-]?(\d+)/i,       // treatment1, treatment_2
        /before|after/i,              // Special handling for before/after
    ];

    for (const pattern of patterns) {
        const match = filename.match(pattern);
        if (match) {
            if (match[1]) return parseInt(match[1]);
            if (match[0].toLowerCase() === 'before') return 0;
            if (match[0].toLowerCase() === 'after') return 9999;
        }
    }
    return null;
};

const findImageSequence = async (matchedImage, allImages) => {
    const directory = path.dirname(matchedImage.path);
    const dirImages = allImages.result.entries
        .filter(img => path.dirname(img.path_lower) === directory)
        .sort((a, b) => {
            const seqA = extractSequenceNumber(a.name);
            const seqB = extractSequenceNumber(b.name);
            if (seqA !== null && seqB !== null) return seqA - seqB;
            return new Date(a.server_modified) - new Date(b.server_modified);
        });

    const matchIndex = dirImages.findIndex(img => 
        img.path_lower === matchedImage.path
    );

    if (matchIndex === -1) return null;

    return {
        current: matchIndex,
        total: dirImages.length,
        before: dirImages.slice(0, matchIndex),
        after: dirImages.slice(matchIndex + 1)
    };
};

// The route handler

app.post('/search/visual', upload.single('image'), async (req, res) => {
    const processingId = crypto.randomUUID();
    const tempFiles = new Set();
    const startTime = Date.now();
    let searchPath = null;

    try {
        if (!req.file && !req.body.imagePath) {
            return res.status(400).json({
                error: 'No image provided',
                code: 'MISSING_IMAGE'
            });
        }

        logger.info('Starting visual search', {
            service: "tatt2awai-bot",
            hasFile: !!req.file,
            filename: req.file?.originalname || req.body.imagePath,
            processingId
        });

        // Handle image input
        let imageData;
        if (req.file) {
            searchPath = req.file.path;
            imageData = await fs.promises.readFile(searchPath);
            tempFiles.add(searchPath);
        } else {
            const fileData = await dropboxManager.downloadFile(req.body.imagePath);
            if (!fileData?.result?.fileBinary) {
                throw new Error('Failed to download image from Dropbox');
            }
            imageData = Buffer.isBuffer(fileData.result.fileBinary) ?
                fileData.result.fileBinary :
                Buffer.from(fileData.result.fileBinary);
            
            searchPath = path.join('uploads', `temp_${Date.now()}_search.jpg`);
            await fs.promises.writeFile(searchPath, imageData);
            tempFiles.add(searchPath);
        }

        // Optimize image before processing
        const optimizedBuffer = await ImageOptimizer.optimizeForSignature(imageData);

        // Ensure initialization
        if (!EnhancedSignatureStore.initialized) {
            throw new Error('Signature store not initialized');
        }

        // Get all images from Dropbox
        const allImages = await dropboxManager.fetchDropboxEntries('');
        if (!allImages?.result?.entries) {
            throw new Error('Failed to fetch images from Dropbox');
        }

        // Find exact match
        const match = await EnhancedSignatureStore.findExactMatch(optimizedBuffer, allImages);

        if (!match) {
            const processingTime = Date.now() - startTime;
            
            // Store search result
            await supabase.from('search_history').insert({
                processing_id: processingId,
                query_image: req.file?.originalname || req.body.imagePath,
                status: 'not_found',
                processing_time: processingTime,
                created_at: new Date().toISOString()
            });

            return res.json({
                success: false,
                message: 'No matching image found',
                processingId,
                metadata: {
                    processingTime,
                    timestamp: new Date().toISOString()
                }
            });
        }

        // Find sequence if match found
        const sequence = await findImageSequence(match, allImages);
        const processingTime = Date.now() - startTime;

        // Store successful search result
        await supabase.from('search_history').insert({
            processing_id: processingId,
            query_image: req.file?.originalname || req.body.imagePath,
            match_path: match.path,
            confidence: match.confidence,
            processing_time: processingTime,
            status: 'found',
            created_at: new Date().toISOString()
        });

        return res.json({
            success: true,
            processingId,
            match: {
                path: match.path,
                confidence: match.confidence,
                similarity: match.score,
                matchType: match.isExact ? 'exact' : 'similar',
                details: match.components
            },
            sequence: sequence ? {
                current: sequence.current,
                total: sequence.total,
                before: sequence.before.map(img => ({
                    path: img.path_lower,
                    modified: img.server_modified
                })),
                after: sequence.after.map(img => ({
                    path: img.path_lower,
                    modified: img.server_modified
                }))
            } : null,
            metadata: {
                processingTime,
                memoryUsage: process.memoryUsage().heapUsed / (1024 * 1024),
                timestamp: new Date().toISOString()
            }
        });

    } catch (error) {
        logger.error('Search error:', {
            service: "tatt2awai-bot",
            error: error.message,
            stack: error.stack,
            processingId
        });

        await supabase.from('search_errors').insert({
            processing_id: processingId,
            error: error.message,
            stack: error.stack,
            created_at: new Date().toISOString()
        });

        return res.status(500).json({
            error: error.message,
            code: 'SEARCH_ERROR',
            processingId
        });

    } finally {
        // Cleanup temp files
        for (const file of tempFiles) {
            try {
                if (fs.existsSync(file)) {
                    await fs.promises.unlink(file);
                    logger.debug('Cleaned up temp file:', {
                        service: "tatt2awai-bot",
                        path: file,
                        processingId
                    });
                }
} catch (cleanupError) {
                logger.warn('Error cleaning up temp file:', {
                    service: "tatt2awai-bot",
                    path: file,
                    error: cleanupError.message,
                    processingId
                });
            }
        }

        // Clear memory
        if (global.gc) {
            global.gc();
        }
    }
});


async function validateAndRetrieveImageData(signature, path) {
    try {
        if (Buffer.isBuffer(signature.imageData)) {
            return signature.imageData;
        }

        const fileData = await dropboxManager.downloadFile(path);
        if (!fileData?.result?.fileBinary) {
            return null;
        }

        return Buffer.isBuffer(fileData.result.fileBinary) ?
            fileData.result.fileBinary :
            Buffer.from(fileData.result.fileBinary);
    } catch (error) {
        logger.error('Failed to retrieve image data:', {
            service: "tatt2awai-bot",
            path,
            error: error.message
        });
        return null;
    }
}

async function cleanupResources(tempFiles, context) {
    for (const file of tempFiles) {
        try {
            if (fs.existsSync(file)) {
                await fs.promises.unlink(file);
                logger.debug('Cleaned up temp file:', { path: file });
            }
        } catch (error) {
            logger.warn('Error cleaning up temp file:', {
                path: file,
                error: error.message
            });
        }
    }
}


// Core processing functions
async function getImageDataFromFile(filePath, context) {
    try {
        // Get original buffer once
        const originalBuffer = await fs.promises.readFile(filePath);
        
        // Process in parallel with new Sharp instances for each operation
        const [
            metadata,
            processed,
            enhanced,
            normalized,
            edge
        ] = await Promise.all([
            sharp(originalBuffer).metadata(),
            sharp(originalBuffer)
                .normalize()
                .removeAlpha()
                .toBuffer(),
            sharp(originalBuffer)
                .normalize()
                .gamma(1.2)
                .modulate({ brightness: 1.05, saturation: 1.1 })
                .toBuffer(),
            sharp(originalBuffer)
                .normalize()
                .removeAlpha()
                .toBuffer(),
            sharp(originalBuffer)
                .greyscale()
                .normalize()
                .toBuffer()
        ]);

        return {
            metadata,
            processed,
            enhanced,
            normalized,
            edge,
            original: originalBuffer
        };
    } catch (error) {
        logger.error('Error in getImageDataFromFile:', {
            error: error.message,
            stack: error.stack,
            filePath
        });
        throw new Error('Failed to process image data: ' + error.message);
    }
}

function getDistributionType(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const q1 = sorted[Math.floor(sorted.length / 4)];
    const q3 = sorted[Math.floor(3 * sorted.length / 4)];
    
    if (q3 - q1 < median * 0.5) return 'uniform';
    if (q3 - q1 > median * 2) return 'scattered';
    return 'normal';
}

function calculateEdgeStats(edgeData) {
    const values = Array.from(edgeData);
    const average = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - average, 2), 0) / values.length;
    
    return {
        average,
        variance,
        distribution: getDistributionType(values)
    };
}


function calculateEdgeStrength(gradients) {
    if (!gradients || !gradients.gradX || !gradients.gradY) {
        return 0;
    }
    
    let totalStrength = 0;
    const length = gradients.gradX.length;
    
    for (let i = 0; i < length; i++) {
        const gx = gradients.gradX[i];
        const gy = gradients.gradY[i];
        totalStrength += Math.sqrt(gx * gx + gy * gy);
    }
    
    return length > 0 ? totalStrength / length : 0;
}

// Edge distribution calculation
function calculateEdgeDistribution(data, width, height) {
    if (!data || !width || !height) {
        return { uniform: false, density: 0 };
    }

    let edgeCount = 0;
    const threshold = 30; // Adjustable threshold for edge detection

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            const gx = Math.abs(data[idx + 1] - data[idx - 1]);
            const gy = Math.abs(data[idx + width] - data[idx - width]);
            
            if (Math.sqrt(gx * gx + gy * gy) > threshold) {
                edgeCount++;
            }
        }
    }

    const density = edgeCount / ((width - 2) * (height - 2));
    const uniform = density > 0.1 && density < 0.5;

    return { uniform, density };
}

// Main edge analysis function
async function analyzeEdges(buffer) {
    try {
        const image = sharp(buffer);
        const { data, info } = await image
            .greyscale()
            .raw()
            .toBuffer({ resolveWithObject: true });

        if (!data || !info.width || !info.height) {
            throw new Error('Invalid image data or dimensions');
        }

        const gradients = calculateGradients(data, info.width, info.height);
        const orientations = calculateOrientations(gradients);
        const distribution = calculateEdgeDistribution(data, info.width, info.height);
        const strength = calculateEdgeStrength(gradients);
        
        return {
            strength,
            orientation: analyzeOrientations(orientations),
            distribution
        };
    } catch (error) {
        logger.error('Edge analysis error:', {
            error: error.message,
            stack: error.stack
        });
        return {
            strength: 0,
            orientation: { dominant: 'unknown', distribution: [], strength: 0 },
            distribution: { uniform: false, density: 0 }
        };
    }
}



function calculateImageComplexity(data) {
    // Simple complexity measure based on value changes
    let changes = 0;
    for (let i = 1; i < data.length; i++) {
        if (Math.abs(data[i] - data[i - 1]) > 10) {
            changes++;
        }
    }
    return changes / data.length;
}

function detectImagePatterns(data, width, height) {
    const patterns = [];
    
    // Detect horizontal patterns
    for (let y = 0; y < height; y++) {
        let repeats = 0;
        for (let x = 1; x < width; x++) {
            if (Math.abs(data[y * width + x] - data[y * width + x - 1]) < 10) {
                repeats++;
            }
        }
        if (repeats > width * 0.8) {
            patterns.push('horizontal');
            break;
        }
    }
    
    // Detect vertical patterns
    for (let x = 0; x < width; x++) {
        let repeats = 0;
        for (let y = 1; y < height; y++) {
            if (Math.abs(data[y * width + x] - data[(y - 1) * width + x]) < 10) {
                repeats++;
            }
        }
        if (repeats > height * 0.8) {
            patterns.push('vertical');
            break;
        }
    }
    
    return patterns;
}

async function analyzeContent(buffer) {
    try {
        const image = sharp(buffer);
        const { data, info } = await image
            .raw()
            .toBuffer({ resolveWithObject: true });

        return {
            complexity: calculateImageComplexity(data),
            brightness: calculateAverageBrightness(data),
            contrast: calculateImageContrast(data),
            patterns: detectImagePatterns(data, info.width, info.height)
        };
    } catch (error) {
        logger.error('Content analysis error:', error);
        return {
            complexity: 0,
            brightness: 0,
            contrast: 0,
            patterns: []
        };
    }
}


async function analyzeImageCharacteristics(imageData) {
    const [
        basicStats,
        edgeAnalysis,
        contentAnalysis
    ] = await Promise.all([
        calculateImageStatistics(imageData.processed),
        analyzeEdges(imageData.edge),
        analyzeContent(imageData.processed)
    ]);

    return {
        type: determineImageType(basicStats, edgeAnalysis),
        quality: assessImageQuality(basicStats),
        content: contentAnalysis,
        statistics: basicStats,
        edges: edgeAnalysis
    };
}

async function validateImageData(imageData) {
    if (!imageData || !imageData.processed || !imageData.metadata) {
        throw new Error('Invalid image data structure');
    }

    // Check image dimensions
    if (imageData.metadata.width > 15000 || imageData.metadata.height > 15000) {
        throw new Error('Image dimensions too large');
    }

    // Check file size
    if (imageData.original.length > 25 * 1024 * 1024) { // 25MB limit
        throw new Error('Image file size too large');
    }

    return true;
}

async function findExactMatches(searchSignature, imageData) {
    const exactMatches = [];
    
    for (const [path, signature] of imageSignatures.entries()) {
        const quickScore = await imageProcessor.calculateQuickScore(searchSignature, signature);
        
        if (quickScore > 0.95) {
            const detailedComparison = await ImageComparison.compareImages(
                imageData.processed,
                signature.imageData
            );

            if (detailedComparison.similarity > 0.98) {
                exactMatches.push({
                    path,
                    signature,
                    confidence: detailedComparison.confidence,
                    similarity: detailedComparison.similarity,
                    matchType: 'exact'
                });
            }
        }
    }

    return exactMatches;
}

async function findContentBasedMatches(features, context) {
    const contentMatches = [];
    const contentThreshold = 0.85;

    for (const [path, signature] of imageSignatures.entries()) {
        try {
            const contentSimilarity = await ImageComparison.compareImageContent(
                features,
                signature.features
            );

            if (contentSimilarity.score > contentThreshold) {
                contentMatches.push({
                    path,
                    signature,
                    confidence: contentSimilarity.confidence,
                    similarity: contentSimilarity.score,
                    matchType: 'content',
                    details: contentSimilarity.details
                });
            }
        } catch (error) {
            logger.warn('Error in content comparison:', {
                path,
                error: error.message
            });
        }
    }

    return contentMatches;
}

async function findVisualSimilarityMatches(imageData, context) {
    const visualMatches = [];
    const visualThreshold = 0.60;
    const dropboxCache = new Map();
    let dropboxAuthenticated = false;

try {
        // Ensure search image is a buffer
        const searchBuffer = Buffer.isBuffer(imageData.processed) ?
            imageData.processed :
            Buffer.from(imageData.processed);

        // Generate hash for search image
        const tempSearchPath = path.join('uploads', `temp_search_${Date.now()}.jpg`);
        try {
            fs.writeFileSync(tempSearchPath, searchBuffer);
            
            // Use promisified version of imageHash
            const generateHash = (size) => new Promise((resolve, reject) => {
                imageHash(tempSearchPath, size, true, (error, hash) => {
                    if (error) reject(error);
                    else resolve({ size, hash });
                });
            });

            const searchHashes = await Promise.all([16, 32, 64].map(size => generateHash(size)));

            const signatures = Array.from(imageSignatures.entries());
            const totalBatches = Math.ceil(signatures.length / BATCH_SIZE);

            logger.info('Starting visual search with details:', {
                service: "tatt2awai-bot",
                totalImages: signatures.length,
                batchSize: BATCH_SIZE,
                totalBatches,
                searchBufferSize: searchBuffer.length,
                hasImageSignatures: signatures.length > 0
            });

            // First pass: quick hash comparison
const potentialMatches = [];
for (const [path, signature] of signatures) {
    if (signature.perceptualHashes) {
        // Compare hashes with tolerance
        const hashSimilarities = searchHashes.map(searchHash => 
            signature.perceptualHashes
                .filter(storedHash => storedHash.size === searchHash.size)
                .map(storedHash => {
                    const hammingDistance = calculateHammingDistance(searchHash.hash, storedHash.hash);
                    // Convert hamming distance to similarity score (0-1)
                    return 1 - (hammingDistance / (searchHash.hash.length * 4));
                })
        ).flat();

        // If any hash similarity is above threshold, consider it a potential match
        if (hashSimilarities.some(similarity => similarity > 0.7)) {
            potentialMatches.push([path, signature]);
        }
    }
}

// Add this helper function at the top level
function calculateHammingDistance(hash1, hash2) {
    let distance = 0;
    const len = Math.min(hash1.length, hash2.length);
    for (let i = 0; i < len; i++) {
        if (hash1[i] !== hash2[i]) distance++;
    }
    return distance;
}

logger.info('Hash comparison details:', {
    service: "tatt2awai-bot",
    totalSignatures: signatures.length,
    signaturesSampleHash: signatures[0]?.[1]?.perceptualHashes?.[0]?.hash?.substring(0, 10),
    searchHashesSample: searchHashes[0]?.hash?.substring(0, 10),
    potentialMatches: potentialMatches.length
});


            // Process batches of potential matches
            const matchBatches = Math.ceil(potentialMatches.length / BATCH_SIZE);
            for (let batchIndex = 0; batchIndex < matchBatches; batchIndex++) {
                const start = batchIndex * BATCH_SIZE;
                const end = Math.min(start + BATCH_SIZE, potentialMatches.length);
                const batch = potentialMatches.slice(start, end);
                
                const batchResults = await Promise.all(
                    batch.map(async ([path, signature]) => {
                        try {
                            if (!signature.imageData) {
                                try {
                                    const fileData = await dropboxManager.downloadFile(path);
                                    if (fileData?.result?.fileBinary) {
                                        signature.imageData = Buffer.isBuffer(fileData.result.fileBinary) ?
                                            fileData.result.fileBinary :
                                            Buffer.from(fileData.result.fileBinary);
                                        imageSignatures.set(path, signature);
                                    }
                                } catch (downloadError) {
                                    logger.warn(`Failed to download image data for ${path}:`, {
                                        service: "tatt2awai-bot",
                                        error: downloadError.message
                                    });
                                    return null;
                                }
                            }

                            if (!signature.imageData) {
                                logger.warn(`No image data available for ${path}`, {
                                    service: "tatt2awai-bot"
                                });
                                return null;
                            }

                            const similarity = await imageProcessor.compareHash(searchBuffer, signature.imageData);
                            
                            if (similarity > 0.8) {
                                return {
                                    path,
                                    similarity,
                                    confidence: similarity,
                                    matchType: 'visual'
                                };
                            }
                            return null;
                        } catch (error) {
                            logger.warn(`Error comparing with ${path}:`, {
                                service: "tatt2awai-bot",
                                error: error.message
                            });
                            return null;
                        }
                    })
                );

                const validMatches = batchResults.filter(Boolean);
                visualMatches.push(...validMatches);

                logger.info('Batch processing results:', {
                    service: "tatt2awai-bot",
                    batchIndex,
                    processedInBatch: batch.length,
                    matchesInBatch: validMatches.length,
                    totalMatchesSoFar: visualMatches.length,
                    progress: `${((batchIndex + 1) / matchBatches * 100).toFixed(1)}%`
                });

                await new Promise(resolve => setTimeout(resolve, 100));
            }

            return visualMatches.sort((a, b) => b.similarity - a.similarity);

        } finally {
            if (fs.existsSync(tempSearchPath)) {
                fs.unlinkSync(tempSearchPath);
            }
        }

    } catch (error) {
        logger.error('Error in visual similarity search:', {
            service: "tatt2awai-bot",
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

// Helper function to get image data
async function getImageDataFromPath(path, cache) {
    if (cache.has(path)) {
        return cache.get(path);
    }

    try {
        const fileData = await dropboxManager.downloadFile(path);
        if (!fileData?.result?.fileBinary) {
            return null;
        }

        const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
            fileData.result.fileBinary :
            Buffer.from(fileData.result.fileBinary);

        cache.set(path, buffer);
        return buffer;
    } catch (error) {
        logger.warn(`Failed to get image data for ${path}:`, {
            service: "tatt2awai-bot",
            error: error.message
        });
        return null;
    }
}

async function processBatchWithLimit(batch, searchBuffer, cache, concurrentLimit) {
    // Process in smaller chunks to limit concurrency
    const results = [];
    for (let i = 0; i < batch.length; i += concurrentLimit) {
        const chunk = batch.slice(i, i + concurrentLimit);
        const chunkResults = await Promise.all(
            chunk.map(([path, signature]) => 
                processSignature(path, signature, searchBuffer, cache)
                    .catch(error => {
                        logger.warn('Error processing signature:', {
                            service: "tatt2awai-bot",
                            path,
                            error: error.message
                        });
                        return null;
                    })
            )
        );
        results.push(...chunkResults.filter(Boolean));
        
        // Rate limiting delay between chunks
        await new Promise(resolve => setTimeout(resolve, RATE_LIMIT_DELAY));
    }
    return results;
}


async function consolidateMatches(exactMatches, contentMatches, visualMatches, context) {
    try {
        // Create a map to track unique matches by path
        const uniqueMatches = new Map();

        // Helper function to add match to map with score aggregation
        const addToMap = (match, type, score) => {
            if (!match.path) return;
            
            if (uniqueMatches.has(match.path)) {
                const existing = uniqueMatches.get(match.path);
                existing.scores.push({ type, score });
                existing.finalScore = Math.max(existing.finalScore, score);
            } else {
                uniqueMatches.set(match.path, {
                    ...match,
                    scores: [{ type, score }],
                    finalScore: score,
                    matchTypes: new Set([type])
                });
            }
        };

        // Add matches from each source
        exactMatches.forEach(match => addToMap(match, 'exact', match.similarity));
        contentMatches.forEach(match => addToMap(match, 'content', match.similarity));
        visualMatches.forEach(match => addToMap(match, 'visual', match.similarity));

        // Convert map to array and calculate final scores
        const consolidatedMatches = Array.from(uniqueMatches.values())
            .map(match => ({
                path: match.path,
                signature: match.signature,
                confidence: Math.max(...match.scores.map(s => s.score)),
                similarity: match.finalScore,
                matchTypes: Array.from(match.matchTypes),
                details: {
                    scores: match.scores,
                    matchCount: match.scores.length
                }
            }));

        // Sort by confidence and similarity
        return consolidatedMatches.sort((a, b) => {
            const scoreDiff = b.similarity - a.similarity;
            if (Math.abs(scoreDiff) > 0.1) return scoreDiff;
            return b.confidence - a.confidence;
        });

    } catch (error) {
        logger.error('Error consolidating matches:', {
            service: "tatt2awai-bot",
            error: error.message,
            stack: error.stack
        });
        return [];
    }
}

function ensureBuffer(data) {
    if (Buffer.isBuffer(data)) {
        return data;
    }
    if (data instanceof Uint8Array) {
        return Buffer.from(data);
    }
    if (typeof data === 'string') {
        return Buffer.from(data, 'binary');
    }
    throw new Error('Cannot convert data to Buffer');
}

async function verifyAndRankMatches(matches, imageData, context) {
    // Perform detailed verification on top matches
    const verifiedMatches = await Promise.all(
        matches.map(async match => {
            try {
                const verification = await ImageComparison.validateMatch(
                    imageData.processed,
                    match.signature.imageData,
                    {
                        type: match.matchType,
                        threshold: getThresholdForType(match.matchType),
                        characteristics: context.imageCharacteristics
                    }
                );

                return {
                    ...match,
                    verificationDetails: verification,
                    finalConfidence: calculateFinalConfidence(
                        match,
                        verification,
                        context
                    )
                };
            } catch (error) {
                logger.warn('Verification failed:', {
                    path: match.path,
                    error: error.message
                });
                return null;
            }
        })
    );

    // Filter out failed verifications and sort by confidence
    return verifiedMatches
        .filter(match => match !== null)
        .sort((a, b) => b.finalConfidence - a.finalConfidence);
}

// Helper functions
function getThresholdForType(matchType) {
    const thresholds = {
        exact: 0.98,
        content: 0.85,
        visual: 0.80,
        partial: 0.75
    };
    return thresholds[matchType] || 0.80;
}

// Add these helper functions to your server.js

async function extractEnhancedFeatures(imageData) {
    try {
        const [
            colorFeatures,
            edgeFeatures,
            textureFeatures,
            shapeFeatures
        ] = await Promise.all([
            extractColorFeatures(imageData.processed),
            extractEdgeFeatures(imageData.edge),
            extractTextureFeatures(imageData.processed),
            extractShapeFeatures(imageData.processed)
        ]);

        return {
            color: colorFeatures,
            edges: edgeFeatures,
            texture: textureFeatures,
            shapes: shapeFeatures,
            metadata: imageData.metadata
        };
    } catch (error) {
        logger.error('Error extracting enhanced features:', {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

async function calculateImageStatistics(buffer) {
    try {
        const image = sharp(buffer);
        const [stats, metadata] = await Promise.all([
            image.stats(),
            image.metadata()
        ]);

        // Calculate basic statistics
        const channelStats = stats.channels.map(channel => ({
            mean: channel.mean,
            stdev: channel.std,
            min: channel.min,
            max: channel.max,
            entropy: calculateEntropy(channel)
        }));

        return {
            channels: channelStats,
            aspectRatio: metadata.width / metadata.height,
            dimensions: {
                width: metadata.width,
                height: metadata.height
            },
            format: metadata.format,
            space: metadata.space,
            density: metadata.density,
            hasAlpha: metadata.hasAlpha,
            isProgressive: metadata.isProgressive
        };
    } catch (error) {
        logger.error('Error calculating image statistics:', {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

async function extractColorFeatures(buffer) {
    const image = sharp(buffer);
    const { data, info } = await image
        .raw()
        .toBuffer({ resolveWithObject: true });

    // Calculate color histogram
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) {
        histogram[data[i]]++;
    }

    // Calculate dominant colors
    const dominantColors = findDominantColors(data);

    // Calculate color moments
    const colorMoments = calculateColorMoments(data);

    return {
        histogram,
        dominantColors,
        moments: colorMoments
    };
}

async function extractEdgeFeatures(buffer) {
    const image = sharp(buffer);
    
    // Generate edge map
    const edgeMap = await image
        .convolve({
            width: 3,
            height: 3,
            kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
        })
        .raw()
        .toBuffer();

    // Calculate edge statistics
    const edgeStats = calculateEdgeStatistics(edgeMap);

    // Detect edge patterns
    const patterns = detectEdgePatterns(edgeMap);

    return {
        edgeMap,
        statistics: edgeStats,
        patterns
    };
}

function calculateWindowMean(data, x, y, size, width) {
    let sum = 0;
    const pixels = size * size;

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const idx = (y + i) * width + (x + j);
            sum += data[idx];
        }
    }

    return sum / pixels;
}

function calculateWindowVariation(data, x, y, size, width, mean) {
    let variation = 0;
    const pixels = size * size;

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const idx = (y + i) * width + (x + j);
            variation += Math.pow(data[idx] - mean, 2);
        }
    }

    return Math.sqrt(variation / pixels);
}



function calculateCoarseness(data, width, height) {
    const windowSizes = [2, 4, 8, 16];
    let maxCoarseness = 0;

    for (const size of windowSizes) {
        let totalVariation = 0;
        const windows = Math.floor(width / size) * Math.floor(height / size);

        for (let y = 0; y < height - size; y += size) {
            for (let x = 0; x < width - size; x += size) {
                const windowMean = calculateWindowMean(data, x, y, size, width);
                const variation = calculateWindowVariation(data, x, y, size, width, windowMean);
                totalVariation += variation;
            }
        }

        const avgVariation = totalVariation / windows;
        maxCoarseness = Math.max(maxCoarseness, avgVariation);
    }

    return maxCoarseness;
}

function getWindowValues(data, x, y, size, width) {
    const values = [];
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const idx = (y + i) * width + (x + j);
            values.push(data[idx]);
        }
    }
    return values;
}

function calculateLocalContrast(values) {
    const max = Math.max(...values);
    const min = Math.min(...values);
    return max - min;
}

function calculateGradients(data, width, height) {
    const gradients = {
        x: new Float32Array((width - 1) * (height - 1)),
        y: new Float32Array((width - 1) * (height - 1))
    };

    for (let y = 0; y < height - 1; y++) {
        for (let x = 0; x < width - 1; x++) {
            const idx = y * width + x;
            gradients.x[idx] = data[idx + 1] - data[idx];
            gradients.y[idx] = data[idx + width] - data[idx];
        }
    }

    return gradients;
}

function calculateGradientAngles(gradients) {
    const angles = new Float32Array(gradients.x.length);
    for (let i = 0; i < gradients.x.length; i++) {
        angles[i] = Math.atan2(gradients.y[i], gradients.x[i]);
    }
    return angles;
}

function calculateAngleHistogram(angles) {
    const bins = 8;
    const histogram = new Array(bins).fill(0);
    const binSize = Math.PI / bins;

    for (const angle of angles) {
        const normalizedAngle = angle + Math.PI;
        const bin = Math.floor(normalizedAngle / binSize) % bins;
        histogram[bin]++;
    }

    const total = histogram.reduce((a, b) => a + b, 0);
    return histogram.map(count => count / total);
}


function calculateTextureContrast(data, width, height) {
    let totalContrast = 0;
    const windows = Math.floor(width / 8) * Math.floor(height / 8);

    for (let y = 0; y < height - 8; y += 8) {
        for (let x = 0; x < width - 8; x += 8) {
            const windowValues = getWindowValues(data, x, y, 8, width);
            const contrast = calculateLocalContrast(windowValues);
            totalContrast += contrast;
        }
    }

    return totalContrast / windows;
}

function calculateDirectionality(data, width, height) {
    const gradients = calculateGradients(data, width, height);
    const angles = calculateGradientAngles(gradients);
    return calculateAngleHistogram(angles);
}

async function extractTextureFeatures(buffer) {
    const image = sharp(buffer);
    const { data, info } = await image
        .greyscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;

    const coarseness = calculateCoarseness(data, width, height);
    const contrast = calculateTextureContrast(data, width, height);
    const directionality = calculateDirectionality(data, width, height);

    // Calculate GLCM
    const glcm = calculateGLCM(data);
    const haralickFeatures = calculateHaralickFeatures(glcm);

    return {
        coarseness,
        contrast,
        directionality,
        glcm,
        haralick: haralickFeatures
    };
}

async function extractShapeFeatures(buffer) {
    const image = sharp(buffer);
    const { data, info } = await image
        .greyscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

    // Extract contours
    const contours = findContours(data, info.width, info.height);

    // Calculate shape descriptors
    const descriptors = calculateShapeDescriptors(contours);

    return {
        contours,
        descriptors,
        moments: calculateHuMoments(contours)
    };
}

function calculateAdaptiveThreshold(data, width, height) {
    // Otsu's method for thresholding
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) {
        histogram[data[i]]++;
    }

    let sum = 0;
    for (let i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }

    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let maxVariance = 0;
    let threshold = 0;
    const total = width * height;

    for (let i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB === 0) continue;

        wF = total - wB;
        if (wF === 0) break;

        sumB += i * histogram[i];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const variance = wB * wF * (mB - mF) * (mB - mF);

        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

function detectEdges(data, width, height) {
    const edges = new Uint8Array(data.length);
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let pixelX = 0;
            let pixelY = 0;

            for (let i = -1; i <= 1; i++) {
                for (let j = -1; j <= 1; j++) {
                    const idx = (y + i) * width + (x + j);
                    const kernelIdx = (i + 1) * 3 + (j + 1);
                    pixelX += data[idx] * sobelX[kernelIdx];
                    pixelY += data[idx] * sobelY[kernelIdx];
                }
            }

            const magnitude = Math.sqrt(pixelX * pixelX + pixelY * pixelY);
            edges[y * width + x] = magnitude > 128 ? 255 : 0;
        }
    }

    return edges;
}


function traceContours(edges, width, height, threshold) {
    const contours = [];
    const visited = new Set();

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (edges[idx] > threshold && !visited.has(idx)) {
                const contour = [];
                followContour(edges, width, height, x, y, contour, visited, threshold);
                if (contour.length > 10) { // Minimum contour size
                    contours.push(contour);
                }
            }
        }
    }

    return contours;
}

function followContour(edges, width, height, startX, startY, contour, visited, threshold) {
    const stack = [[startX, startY]];
    const directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

    while (stack.length > 0) {
        const [x, y] = stack.pop();
        const idx = y * width + x;

        if (visited.has(idx)) continue;
        visited.add(idx);
        contour.push([x, y]);

        for (const [dx, dy] of directions) {
            const newX = x + dx;
            const newY = y + dy;

            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                const newIdx = newY * width + newX;
                if (edges[newIdx] > threshold && !visited.has(newIdx)) {
                    stack.push([newX, newY]);
                }
            }
        }
    }
}

function calculateCircularity(contour) {
    const area = calculateContourArea(contour);
    const perimeter = calculateContourPerimeter(contour);
    return (4 * Math.PI * area) / (perimeter * perimeter);
}

function calculateCompactness(contour) {
    const area = calculateContourArea(contour);
    const perimeter = calculateContourPerimeter(contour);
    return Math.sqrt(4 * Math.PI * area) / perimeter;
}

function calculateBoundingBox(contour) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const [x, y] of contour) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }

    return {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY
    };
}

function calculateRawMoments(contour) {
    const moments = Array(4).fill().map(() => Array(4).fill(0));
    
    for (const [x, y] of contour) {
        for (let p = 0; p < 4; p++) {
            for (let q = 0; q < 4; q++) {
                moments[p][q] += Math.pow(x, p) * Math.pow(y, q);
            }
        }
    }
    
    return moments;
}

function calculateCentralMoments(contour, rawMoments) {
    const centerX = rawMoments[1][0] / rawMoments[0][0];
    const centerY = rawMoments[0][1] / rawMoments[0][0];
    const centralMoments = Array(4).fill().map(() => Array(4).fill(0));
    
    for (const [x, y] of contour) {
        const xc = x - centerX;
        const yc = y - centerY;
        for (let p = 0; p < 4; p++) {
            for (let q = 0; q < 4; q++) {
                centralMoments[p][q] += Math.pow(xc, p) * Math.pow(yc, q);
            }
        }
    }
    
    return centralMoments;
}

function calculateInvariantMoments(centralMoments) {
    const n20 = centralMoments[2][0] / Math.pow(centralMoments[0][0], 2);
    const n02 = centralMoments[0][2] / Math.pow(centralMoments[0][0], 2);
    const n11 = centralMoments[1][1] / Math.pow(centralMoments[0][0], 2);
    const n30 = centralMoments[3][0] / Math.pow(centralMoments[0][0], 2.5);
    const n03 = centralMoments[0][3] / Math.pow(centralMoments[0][0], 2.5);
    const n21 = centralMoments[2][1] / Math.pow(centralMoments[0][0], 2.5);
    const n12 = centralMoments[1][2] / Math.pow(centralMoments[0][0], 2.5);

    return [
        n20 + n02,
        Math.pow(n20 - n02, 2) + 4 * Math.pow(n11, 2),
        Math.pow(n30 - 3 * n12, 2) + Math.pow(3 * n21 - n03, 2),
        Math.pow(n30 + n12, 2) + Math.pow(n21 + n03, 2),
        (n30 - 3 * n12) * (n30 + n12) * (Math.pow(n30 + n12, 2) - 3 * Math.pow(n21 + n03, 2)) +
        (3 * n21 - n03) * (n21 + n03) * (3 * Math.pow(n30 + n12, 2) - Math.pow(n21 + n03, 2)),
        (n20 - n02) * (Math.pow(n30 + n12, 2) - Math.pow(n21 + n03, 2)) +
        4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03) * (n30 + n12) * (Math.pow(n30 + n12, 2) - 3 * Math.pow(n21 + n03, 2)) -
        (n30 - 3 * n12) * (n21 + n03) * (3 * Math.pow(n30 + n12, 2) - Math.pow(n21 + n03, 2))
    ];
}

function calculateEntropyFromChannel(channel) {
    // Create a normalized histogram from channel statistics
    const histogram = new Float32Array(256);
    const mean = channel.mean;
    const std = channel.std || 1;

    // Create a Gaussian distribution based on mean and std
    for (let i = 0; i < 256; i++) {
        const z = (i - mean) / std;
        histogram[i] = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * std);
    }

    // Normalize histogram
    const sum = histogram.reduce((a, b) => a + b, 0);
    for (let i = 0; i < 256; i++) {
        histogram[i] /= sum;
    }

    // Calculate entropy
    let entropy = 0;
    for (let i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            entropy -= histogram[i] * Math.log2(histogram[i]);
        }
    }

    return entropy;
}

function calculateChannelStatistics(channel) {
    return {
        mean: channel.mean,
        std: channel.std,
        min: channel.min,
        max: channel.max,
        entropy: calculateEntropyFromChannel(channel)
    };
}


function calculateContourArea(contour) {
    let area = 0;
    for (let i = 0; i < contour.length; i++) {
        const j = (i + 1) % contour.length;
        area += contour[i][0] * contour[j][1];
        area -= contour[j][0] * contour[i][1];
    }
    return Math.abs(area) / 2;
}


function findContours(data, width, height) {
    const threshold = calculateAdaptiveThreshold(data, width, height);
    const edges = detectEdges(data, width, height);
    return traceContours(edges, width, height, threshold);
}

function calculateContourPerimeter(contour) {
    let perimeter = 0;
    for (let i = 0; i < contour.length; i++) {
        const j = (i + 1) % contour.length;
        const dx = contour[j][0] - contour[i][0];
        const dy = contour[j][1] - contour[i][1];
        perimeter += Math.sqrt(dx * dx + dy * dy);
    }
    return perimeter;
}

function calculateShapeDescriptors(contours) {
    return contours.map(contour => ({
        area: calculateContourArea(contour),
        perimeter: calculateContourPerimeter(contour),
        circularity: calculateCircularity(contour),
        compactness: calculateCompactness(contour),
        boundingBox: calculateBoundingBox(contour)
    }));
}

function calculateHuMoments(contours) {
    return contours.map(contour => {
        const moments = calculateRawMoments(contour);
        const centralMoments = calculateCentralMoments(contour, moments);
        return calculateInvariantMoments(centralMoments);
    });
}

// Helper functions for feature extraction
function calculateEntropy(channel) {
    if (!channel || typeof channel.mean === 'undefined' || 
        typeof channel.std === 'undefined' || 
        typeof channel.min === 'undefined' || 
        typeof channel.max === 'undefined') {
        return 0;
    }

    // Create histogram using normal distribution approximation
    const bins = 256;
    const histogram = new Float32Array(bins);
    const mean = channel.mean;
    const std = channel.std || 1; // Prevent division by zero
    
    // Generate normalized histogram using Gaussian distribution
    for (let i = 0; i < bins; i++) {
        const x = i / (bins - 1); // Normalize to [0, 1]
        const z = (x - mean / 255) / (std / 255);
        histogram[i] = Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * (std / 255));
    }
    
    // Normalize histogram to sum to 1
    const sum = histogram.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;
    
    for (let i = 0; i < bins; i++) {
        histogram[i] /= sum;
    }
    
    // Calculate entropy
    let entropy = 0;
    for (let i = 0; i < bins; i++) {
        if (histogram[i] > 0) {
            entropy -= histogram[i] * Math.log2(histogram[i]);
        }
    }
    
    return entropy;
}



function findDominantColors(data) {
    const colorCounts = new Map();
    
    // Count color occurrences
    for (let i = 0; i < data.length; i += 3) {
        const color = `${data[i]},${data[i + 1]},${data[i + 2]}`;
        colorCounts.set(color, (colorCounts.get(color) || 0) + 1);
    }

    // Sort and get top colors
    return Array.from(colorCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([color, count]) => ({
            rgb: color.split(',').map(Number),
            frequency: count / (data.length / 3)
        }));
}

function calculateColorMoments(data) {
    const channels = [[], [], []];
    
    // Separate channels
    for (let i = 0; i < data.length; i += 3) {
        channels[0].push(data[i]);     // R
        channels[1].push(data[i + 1]); // G
        channels[2].push(data[i + 2]); // B
    }

    // Calculate moments for each channel
    return channels.map(channel => ({
        mean: calculateMean(channel),
        standardDeviation: calculateStandardDeviation(channel),
        skewness: calculateSkewness(channel)
    }));
}

function calculateGLCM(data) {
    // Simplified GLCM calculation
    const glcm = Array(16).fill().map(() => Array(16).fill(0));
    const width = Math.sqrt(data.length);

    for (let i = 0; i < data.length - 1; i++) {
        if ((i + 1) % width !== 0) { // Skip last pixel in each row
            const value1 = Math.floor(data[i] / 16);
            const value2 = Math.floor(data[i + 1] / 16);
            glcm[value1][value2]++;
            glcm[value2][value1]++; // Symmetric GLCM
        }
    }

    // Normalize GLCM
    const sum = glcm.reduce((a, row) => 
        a + row.reduce((b, val) => b + val, 0), 0);

    return glcm.map(row => 
        row.map(val => val / sum)
    );
}

function calculateHaralickFeatures(glcm) {
    // Calculate basic Haralick features
    const contrast = calculateGLCMContrast(glcm);
    const correlation = calculateGLCMCorrelation(glcm);
    const energy = calculateGLCMEnergy(glcm);
    const homogeneity = calculateGLCMHomogeneity(glcm);

    return {
        contrast,
        correlation,
        energy,
        homogeneity
    };
}

function detectEdgePatterns(edgeMap) {
    const patterns = {
        horizontal: 0,
        vertical: 0,
        diagonal: 0,
        strength: 0
    };

    // Implement edge pattern detection logic here
    
    return patterns;
}

function calculateEdgeStatistics(edgeMap) {
    const stats = {
        mean: 0,
        variance: 0,
        strength: 0,
        density: 0
    };

    // Calculate basic edge statistics
    const mean = edgeMap.reduce((sum, val) => sum + val, 0) / edgeMap.length;
    stats.mean = mean;
    
    stats.variance = edgeMap.reduce((sum, val) => 
        sum + Math.pow(val - mean, 2), 0) / edgeMap.length;
    
    stats.strength = Math.sqrt(stats.variance);
    stats.density = edgeMap.filter(val => val > mean).length / edgeMap.length;

    return stats;
}

// Helper functions for calculations
function calculateMean(array) {
    return array.reduce((sum, val) => sum + val, 0) / array.length;
}

function calculateStandardDeviation(array) {
    const mean = calculateMean(array);
    const variance = array.reduce((sum, val) => 
        sum + Math.pow(val - mean, 2), 0) / array.length;
    return Math.sqrt(variance);
}

function calculateSkewness(array) {
    const mean = calculateMean(array);
    const std = calculateStandardDeviation(array);
    return array.reduce((sum, val) => 
        sum + Math.pow((val - mean) / std, 3), 0) / array.length;
}

// Determine image type based on analysis
function determineImageType(basicStats, edgeAnalysis) {
    const type = {
        isScreenshot: false,
        hasText: false,
        isPhotograph: false,
        isGraphic: false,
        confidence: 0
    };

    // Implement image type detection logic
    // This should be customized based on your specific needs

    return type;
}

function assessImageQuality(basicStats) {
    return {
        sharpness: calculateSharpnessScore(basicStats),
        noise: calculateNoiseScore(basicStats),
        contrast: calculateContrastScore(basicStats),
        colorQuality: calculateColorQualityScore(basicStats),
        overall: 0 // Calculate weighted average of above scores
    };
}

// Add these quality assessment helper functions
function calculateSharpnessScore(stats) {
    // Implement sharpness scoring logic
    return 0.8; // Placeholder
}

function calculateNoiseScore(stats) {
    // Implement noise scoring logic
    return 0.8; // Placeholder
}

function calculateContrastScore(stats) {
    // Implement contrast scoring logic
    return 0.8; // Placeholder
}

function calculateColorQualityScore(stats) {
    // Implement color quality scoring logic
    return 0.8; // Placeholder
}

function calculateFinalConfidence(match, verification, context) {
    const weights = {
        similarity: 0.4,
        verification: 0.3,
        context: 0.2,
        quality: 0.1
    };

    return (
        match.similarity * weights.similarity +
        verification.confidence * weights.verification +
        (context.imageCharacteristics.quality.score * weights.quality) +
        (verification.contextScore * weights.context)
    );
}


// OpenAI Chat Endpoint
app.post('/chat', async (req, res) => {
    try {
        const { message, userId } = req.body;
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        const userUUID = generateUUID(userId || 'default-user');
        const assistant = await getOrCreateAssistant();
        const threadId = await getOrCreateThread(userId || 'default-user');

        // Store chat in Supabase
        await supabase
            .from('chat_history')
            .insert({
                user_id: userUUID,
                thread_id: threadId,
                message_type: 'user',
                content: message
            });

        // Add message to thread
        await openai.beta.threads.messages.create(threadId, {
            role: "user",
            content: message
        });

        // Run assistant
        const run = await openai.beta.threads.runs.create(threadId, {
            assistant_id: assistant.id
        });

        // Wait for completion
        const startTime = Date.now();
        const timeout = 30000;
        let assistantResponse = null;
        
        while (Date.now() - startTime < timeout) {
            const status = await openai.beta.threads.runs.retrieve(threadId, run.id);
            
            if (status.status === 'completed') {
                const messages = await openai.beta.threads.messages.list(threadId);
                assistantResponse = messages.data[0].content[0].text.value;
                
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
        logger.error('Chat error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Server Setup
const httpsOptions = {
    key: fs.readFileSync(process.env.SSL_KEY_PATH),
    cert: fs.readFileSync(process.env.SSL_CERT_PATH),
    secureProtocol: 'TLSv1_2_method',
    rejectUnauthorized: false,
    requestCert: false,
    agent: false,
};

const httpServer = http.createServer(app);
const httpsServer = https.createServer(httpsOptions, app);

async function initializeSignatures() {
    try {
        const allImages = await dropboxManager.fetchDropboxEntries('');
        if (!allImages?.result?.entries) {
            throw new Error('Failed to fetch images from Dropbox');
        }

        await EnhancedSignatureStore.initialize(allImages.result.entries);  // FIXED
        return true;
    } catch (error) {
        logger.error('Failed to initialize signatures:', error);
        throw error;
    }
}

// Start the server
if (require.main === module) {
    initializeWithRetry().catch(error => {
        logger.error('Fatal error during server initialization:', {
            service: "tatt2awai-bot",
            error: error.message,
            stack: error.stack
        });
        process.exit(1);
    });
}

async function initializeWithRetry(maxAttempts = 3) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            await startServer();
            return;
        } catch (error) {
            if (attempt === maxAttempts - 1) {
                logger.error('Server initialization failed after max retries', {
                    service: "tatt2awai-bot",
                    error: error.message,
                    attempts: maxAttempts
                });
                throw error;
            }

            const delay = Math.pow(2, attempt) * 10000; // Exponential backoff
            logger.info(`Retrying initialization in ${delay/1000} seconds...`, {
                service: "tatt2awai-bot",
                attempt: attempt + 1,
                maxAttempts
            });
            
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

async function validateAndInitialize() {
    try {
        const dropboxEntries = await dropboxManager.fetchDropboxEntries('');
        let validCount = 0;
        let invalidCount = 0;
        const invalidFiles = [];

        logger.info('Starting comprehensive image validation...');

        for (const entry of dropboxEntries.result.entries) {
            if (!entry.path_lower.match(/\.(jpg|jpeg|png|webp)$/i)) continue;

            try {
                const fileData = await dropboxManager.downloadFile(entry.path_lower);
                if (!fileData?.result?.fileBinary) continue;

                const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                    fileData.result.fileBinary :
                    Buffer.from(fileData.result.fileBinary);

                const validation = await EnhancedValidation.validateImageContent(buffer);

                if (validation.isValidImage) {
                    validCount++;
                } else {
                    invalidCount++;
                    invalidFiles.push({
                        path: entry.path_lower,
                        warnings: validation.warnings
                    });
                }

                if ((validCount + invalidCount) % 100 === 0) {
                    logger.info(`Validated ${validCount + invalidCount} images. Valid: ${validCount}, Invalid: ${invalidCount}`);
                }

            } catch (error) {
                logger.error(`Failed to validate ${entry.path_lower}:`, error);
                invalidCount++;
                invalidFiles.push({
                    path: entry.path_lower,
                    error: error.message
                });
            }
        }

        if (invalidFiles.length > 0) {
            logger.warn('Found invalid images:', invalidFiles);
            
            // Optional: prevent server start if there are invalid images
            // throw new Error('Invalid images detected');
        }

        return {
            valid: validCount,
            invalid: invalidCount,
            invalidFiles
        };
    } catch (error) {
        logger.error('Validation failed:', error);
        throw error;
    }
}

async function startServer() {
    try {
        logger.info('Starting server initialization...', {
            service: "tatt2awai-bot"
        });

        // Step 1: Ensure Dropbox authentication
        await dropboxManager.ensureAuth();
        logger.info('Dropbox authentication successful', {
            service: "tatt2awai-bot"
        });

        // Step 2: Fetch all Dropbox entries
        const dropboxEntries = await dropboxManager.fetchDropboxEntries('');
        if (!dropboxEntries?.result?.entries) {
            throw new Error('Failed to fetch Dropbox entries');
        }

        // Filter for image files only
        const imageEntries = dropboxEntries.result.entries.filter(entry =>
            ['.jpg', '.jpeg', '.png', '.webp'].some(ext =>
                entry.path_lower.endsWith(ext)
            )
        );

        logger.info('Fetched Dropbox entries', {
            service: "tatt2awai-bot",
            totalEntries: dropboxEntries.result.entries.length,
            imageFiles: imageEntries.length
        });

        // Step 3: Initialize signature store with full entries list
        await EnhancedSignatureStore.initialize(imageEntries);

        // Step 4: Verify signature count
        const storeStatus = EnhancedSignatureStore.getStatus();
        if (storeStatus.signatureCount < imageEntries.length) {
            logger.warn('Signature count mismatch detected. Starting signature generation...', {
                service: "tatt2awai-bot",
                signatureCount: storeStatus.signatureCount,
                expectedCount: imageEntries.length
            });

            // Process missing entries
            const missingCount = imageEntries.length - storeStatus.signatureCount;
            logger.info(`Generating signatures for ${missingCount} missing entries...`);
            
            // The store will automatically process missing entries
            await EnhancedSignatureStore.initialize(imageEntries);

            // Verify final status
            const finalStatus = EnhancedSignatureStore.getStatus();
            logger.info('Signature generation complete', {
                service: "tatt2awai-bot",
                finalSignatureCount: finalStatus.signatureCount,
                expectedCount: imageEntries.length,
                memoryUsage: finalStatus.memoryUsage
            });
        }

        // Step 5: Start servers
        httpsServer.listen(HTTPS_PORT, '0.0.0.0', () => {
            logger.info(`HTTPS server running on port ${HTTPS_PORT}`, {
                service: "tatt2awai-bot"
            });
        });

        httpServer.listen(HTTP_PORT, '0.0.0.0', () => {
            logger.info(`HTTP server running on port ${HTTP_PORT}`, {
                service: "tatt2awai-bot"
            });
        });

        // Add shutdown handler for clean exit
        process.on('SIGTERM', async () => {
            logger.info('Received SIGTERM signal. Starting graceful shutdown...', {
                service: "tatt2awai-bot"
            });
            
            await EnhancedSignatureStore.cleanup();
            process.exit(0);
        });

        process.on('SIGINT', async () => {
            logger.info('Received SIGINT signal. Starting graceful shutdown...', {
                service: "tatt2awai-bot"
            });
            
            await EnhancedSignatureStore.cleanup();
            process.exit(0);
        });

    } catch (error) {
        logger.error('Failed to start server:', {
            service: "tatt2awai-bot",
            error: error.message,
            stack: error.stack
        });
        process.exit(1);
    }
}

// Export necessary components
// Export necessary components
module.exports = {
    app,
    httpServer,
    httpsServer,
    startServer,
initializeWithRetry, 
   initializeImageCache,
    isInitialized: () => isInitialized,
    getInitializationProgress: () => ({ ...initializationProgress }),
    imageAnalysis: {
        calculateImageStatistics,
        extractEnhancedFeatures,
        analyzeImageCharacteristics,
        calculateGLCMContrast,
        calculateGLCMCorrelation,
        calculateGLCMEnergy,
        calculateGLCMHomogeneity
    },
    ImageComparison: {
        ...ImageComparison,
        analyzeImageContent: ImageComparison.analyzeImageContent
    }
};
