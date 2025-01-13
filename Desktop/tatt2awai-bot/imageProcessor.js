// Import required modules
require('dotenv').config();

const fs = require('fs');
const path = require('path');
const vision = require('@google-cloud/vision');
const sharp = require('sharp');
const crypto = require('crypto');
const logger = require('./logger');
const os = require('os');
const { v4: uuidv4 } = require('uuid');
const { createClient } = require('@supabase/supabase-js');
const dropboxManager = require('./dropboxManager');
const pLimit = require('p-limit');
const { LRUCache } = require('lru-cache');
const EnhancedSignatureGenerator = require('./enhanced-signature-generator');
const EnhancedSignatureStore = require('./enhanced-store');
const { ImageOptimizer, MemoryManager } = require('./performance-optimizations');



// Initialize caches and limiters
const fileCache = new LRUCache({
    max: 500, // Cache last 500 files
    maxAge: 1000 * 60 * 5, // 5 minute cache
    updateAgeOnGet: true
});

const limit = pLimit(5);

// Initialize comparison libraries
const comparisonLibs = {
    ssim: null,
    pixelmatch: null
};

// Core Constants
const SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];

const SEARCH_CONSTANTS = {
    PARALLEL_BATCHES: 5,
    BATCH_SIZE: 100,
    HASH_SIZES: {
        QUICK: 16,
        MEDIUM: 32,
        DETAILED: 64
    },
    THRESHOLDS: {
        QUICK_MATCH: 0.75,
        SECONDARY_MATCH: 0.8,
        GOOD_MATCH: 0.9,
        PERFECT_MATCH: 0.95
    },
    TIMEOUTS: {
        DOWNLOAD: 8000,  // Increased timeout
        RETRY_DELAY: 1000
    },
DOWNLOAD: {
        MAX_CONCURRENT: 3,
        TIMEOUT: 10000,
        RETRY_DELAYS: [1000, 2000, 4000], // Exponential backoff
        MAX_RETRIES: 2
    },
 STAGES: {
        QUICK: {
            THRESHOLD: 0.8,
            MAX_MATCHES: 50
        },
        DETAILED: {
            THRESHOLD: 0.9,
            MAX_MATCHES: 5
        },
        FINAL: {
            THRESHOLD: 0.95
        }
    }
};

// Previous imports and initial constants remain the same...

const CACHE_SETTINGS = {
    DURATION: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
    ANALYSIS_DURATION: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
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

const IMAGE_PROCESSING = {
    MAX_SIZE: 4096,
    QUALITY: 90,
    MIN_SIZE: 300,
    OPTIMAL_DPI: 300,
    MIN_IMAGE_SIZE: 100,
    MAX_IMAGE_SIZE: 4096
};

const ENHANCED_SETTINGS = {
    PREPROCESSING: {
        MAX_SIZE: 4096,
        MIN_SIZE: 100,
        QUALITY: 90
    },
    COMPARISON: {
        EXACT_THRESHOLD: 0.95,
        SIMILAR_THRESHOLD: 0.85,
        QUICK_THRESHOLD: 0.80
    }
};


const SEQUENCE_PATTERNS = [
    /before|after|during/i,
    /removal/i,
    /t[0-9]+/i,
    /session[0-9]+/i,
    /treatment[0-9]+/i,
    /\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b/
];

const MATCH_CONSTANTS = {
    BATCH_SIZE: 500,
    CONCURRENT_LIMIT: 10,
    THRESHOLDS: {
        ASPECT_RATIO: 0.1,
        QUICK_MATCH: 0.8,
        DETAILED_MATCH: 0.95,
        PERFECT_MATCH: 0.98
    },
    DIMENSIONS: {
        QUICK: 64,
        DETAILED: 256,
        FINAL: 1024
    }
};

MATCH_CONSTANTS.TIMEOUT = {
    DOWNLOAD: 20000,     // 20 seconds for download
    COMPARISON: 10000,   // 10 seconds for comparison
    BATCH: 120000       // 2 minutes per batch
};

const SEQUENCE_KEYWORDS = [
    'before', 'after', 'during',
    'removal', 'treatment', 'session',
    'healing', 'progress', 'fading',
    'result', 'comparison', 't1', 't2', 't3'
];

const TATTOO_KEYWORDS = new Set([
    'tattoo', 'ink', 'skin', 'art', 'removal', 
    'treatment', 'before', 'after', 'session',
    'laser', 'fading', 'healing', 'progress', 'design', 'mark', 
    'pigment', 'color', 'drawing', 'pattern', 'permanent', 
    'body', 'decoration', 'artwork', 'line', 'black', 
    'dark', 'image', 'design'
]);

const COSMETIC_TATTOO_PATTERNS = {
    KEYWORDS: [
        'permanent makeup', 'microblading', 'cosmetic tattoo', 
        'eyebrow tattoo', 'micropigmentation', 'permanent cosmetics', 
        'brow tattoo', 'powder brows', 'ombré brows', 'microshading'
    ],
    FACIAL_FEATURES: [
        'eyebrow', 'brow', 'eyeliner', 'lip liner', 
        'lip blend', 'lip color', 'beauty mark', 'facial'
    ],
    LOCATIONS: {
        eyebrows: ['brow', 'eyebrow', 'forehead'],
        eyes: ['eyeliner', 'eye line', 'lash line'],
        lips: ['lip liner', 'lip color', 'lip blend', 'lips']
    }
};

const BODY_PART_INDICATORS = {
    HIGH_LIKELIHOOD: ['arm', 'forearm', 'wrist', 'ankle', 'back', 'chest', 'shoulder', 'neck'],
    MEDIUM_LIKELIHOOD: ['finger', 'hand', 'foot', 'leg', 'thigh', 'stomach', 'hip'],
    SKIN_INDICATORS: ['flesh', 'skin', 'stain', 'mark', 'spot'],
    CONDITION_INDICATORS: ['scar', 'healing', 'itch', 'redness', 'discoloration', 'fading']
};

const ENHANCED_PATTERNS = {
    TATTOO_KEYWORDS: new Set([
        'tattoo', 'ink', 'inking', 'inked', 'body art', 'permanent',
        'removal', 'removed', 'removing', 'fading', 'faded',
        'treatment', 'healing', 'healed', 'session',
        'before', 'after', 'during', 'progress', 'progression',
        'results', 'comparison', 'change', 'difference',
        't1', 't2', 't3', 't4', 't5', 
        'treatment 1', 'treatment 2', 'treatment 3',
        'session 1', 'session 2', 'session 3',
        'week', 'month', 'year', 'day',
        'fresh', 'healing', 'healed', 'complete', 'final'
    ]),
    BODY_LOCATIONS: new Set([
        'arm', 'forearm', 'bicep', 'shoulder', 'upper arm', 'lower arm',
        'leg', 'thigh', 'calf', 'ankle', 'foot',
        'chest', 'back', 'torso', 'stomach', 'abdomen',
        'neck', 'face', 'hand', 'wrist', 'finger',
        'hip', 'ribs', 'side', 'spine'
    ]),
    SKIN_TONES: [
        { min: [141, 85, 36], max: [255, 219, 172], name: 'light' },
        { min: [89, 47, 42], max: [250, 190, 175], name: 'medium' },
        { min: [63, 38, 33], max: [195, 140, 110], name: 'dark' }
    ],
    FILE_PATTERNS: {
        SEQUENCE: /(?:before|after|during|t\d|session\d|treatment\d)/i,
        DATE: /\d{1,2}[-./]\d{1,2}[-./]\d{2,4}/,
        PROGRESS: /(?:progress|update|change|result|comparison)/i
    },
    INK_COLORS: {
        BLACK: { max: 60, channel: 'all' },
        BLUE: { min: [0, 0, 100], max: [100, 100, 255] },
        RED: { min: [150, 0, 0], max: [255, 100, 100] },
        GREEN: { min: [0, 100, 0], max: [100, 255, 100] }
    }
};

const ANALYSIS_THRESHOLDS = {
    TATTOO_CONFIDENCE: 0.7,
    COLOR_MATCH: 0.85,
    SEQUENCE_SIMILARITY: 0.80,
    MIN_SEQUENCE_IMAGES: 2,
    MAX_SEQUENCE_GAP_DAYS: 90,
    SKIN_REDNESS_THRESHOLD: 0.6,
    EDGE_SHARPNESS: 0.65,
    MIN_COLOR_PROMINENCE: 0.05,
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

// Initialize Supabase client with error handling
let supabase;
try {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_KEY) {
        throw new Error('Missing required Supabase environment variables');
    }
    supabase = createClient(
        process.env.SUPABASE_URL,
        process.env.SUPABASE_KEY
    );
} catch (error) {
    logger.error('Supabase initialization failed:', {
        error: error.message,
        missingUrl: !process.env.SUPABASE_URL,
        missingKey: !process.env.SUPABASE_KEY
    });
    // Initialize with null client to prevent crashes
    supabase = {
        from: () => ({
            select: () => ({ data: null }),
            insert: () => ({ data: null }),
            upsert: () => ({ data: null })
        })
    };
}

// Initialize Vision API client
const visionClient = new vision.ImageAnnotatorClient({
    keyFilename: path.join(__dirname, process.env.GOOGLE_APPLICATION_CREDENTIALS),
});

class DownloadQueue {
    constructor(maxConcurrent = 3) {
        this.queue = [];
        this.running = 0;
        this.maxConcurrent = maxConcurrent;
    }

    async add(fn) {
        return new Promise((resolve, reject) => {
            const task = {
                fn,
                resolve,
                reject
            };

            this.queue.push(task);
            this.process();
        });
    }

    async process() {
        if (this.running >= this.maxConcurrent || this.queue.length === 0) {
            return;
        }

        this.running++;
        const task = this.queue.shift();

        try {
            const result = await task.fn();
            task.resolve(result);
        } catch (error) {
            task.reject(error);
        } finally {
            this.running--;
            this.process();
        }
    }
}

class ImageHasher {
    static async generateHash(buffer, size = 16) {
        try {
            const processedBuffer = await sharp(buffer)
                .resize(size, size, { fit: 'contain' })
                .greyscale()
                .normalize()
                .raw()
                .toBuffer();

            const pixels = Array.from(processedBuffer);
            const mean = pixels.reduce((sum, val) => sum + val, 0) / pixels.length;
            
            // Binary hash generation
            const hash = pixels.map(p => p >= mean ? '1' : '0').join('');
            return hash;
        } catch (error) {
            throw new Error(`Hash generation failed: ${error.message}`);
        }
    }

    static async generateMultiHash(buffer) {
        const hashes = await Promise.all([16, 32, 64].map(size => 
            this.generateHash(buffer, size)
        ));
        return hashes;
    }

    static calculateSimilarity(hash1, hash2) {
        if (hash1.length !== hash2.length) return 0;
        let diff = 0;
        for (let i = 0; i < hash1.length; i++) {
            if (hash1[i] !== hash2[i]) diff++;
        }
        return 1 - (diff / hash1.length);
    }
}

// Core cache implementation for image processing
class ImageCache {
    constructor() {
        this.cache = new Map();
        this.maxSize = 1000;
        this.cleanupInterval = setInterval(() => this.cleanup(), 3600000); // Clean every hour
    }

    async get(key, generator) {
        if (this.cache.has(key)) {
            const entry = this.cache.get(key);
            if (Date.now() - entry.timestamp < 3600000) {
                return entry.data;
            }
        }
        const data = await generator();
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
        return data;
    }

    cleanup() {
        const now = Date.now();
        for (const [key, entry] of this.cache.entries()) {
            if (now - entry.timestamp > 3600000) {
                this.cache.delete(key);
            }
        }
    }
}

// Initialize the analysis cache
const analysisCache = new ImageCache();

// Signature Generator for image matching and comparison
class ImageSignatureGenerator {
    static async generate(imageData) {
        const tempPath = path.join(os.tmpdir(), `${uuidv4()}.jpg`);
        try {
            // Enhanced preprocessing
            const preprocessed = await sharp(imageData)
                .trim({ background: 'white', threshold: 10 })
                .normalize()
                .gamma(1.2)
                .modulate({ brightness: 1.05, saturation: 1.1 })
                .removeAlpha()
                .jpeg({ quality: 100 })
                .toBuffer();

            await fs.promises.writeFile(tempPath, preprocessed);

            // Generate multiple perceptual hashes
            const hashes = await Promise.all([16, 32, 64].map(size =>
                new Promise((resolve, reject) => {
                    imageHash(tempPath, size, true, (error, hash) => {
                        if (error) reject(error);
                        else resolve({ size, hash });
                    });
                })
            ));

            // Enhanced metadata
            const [stats, metadata] = await Promise.all([
                sharp(preprocessed).stats(),
                sharp(preprocessed).metadata()
            ]);

            // Generate color histogram
            const colorHistogram = await this.generateDetailedColorHistogram(preprocessed);

            // Generate edge signature
            const edgeSignature = await this.generateAdvancedEdgeSignature(preprocessed);

            // Generate binary hash
            const binarySignature = await this.generateBinarySignature(preprocessed);

            return {
                metadata: {
                    dimensions: {
                        width: metadata.width,
                        height: metadata.height,
                        aspectRatio: metadata.width / metadata.height
                    },
                    format: 'jpeg',
                    space: metadata.space
                },
                colorMetrics: {
                    means: stats.channels.map(c => c.mean),
                    medians: stats.channels.map(c => c.median),
                    stds: stats.channels.map(c => c.std),
                    histogram: colorHistogram
                },
                perceptualHashes: hashes,
                edgeSignature,
                binarySignature,
                signatureId: crypto.createHash('sha256')
                    .update(preprocessed)
                    .digest('hex'),
                generatedAt: new Date().toISOString()
            };
        } catch (error) {
            logger.error('Error generating image signature:', error);
            throw error;
        } finally {
            try {
                if (fs.existsSync(tempPath)) {
                    fs.unlinkSync(tempPath);
                }
            } catch (cleanupError) {
                logger.warn('Error cleaning up temp file:', cleanupError);
            }
        }
    }
static async generateDetailedColorHistogram(buffer) {
        const image = sharp(buffer);
        const { data, info } = await image
            .raw()
            .toBuffer({ resolveWithObject: true });

        const bins = 32;
        const histogram = new Array(bins * bins * bins).fill(0);

        for (let i = 0; i < data.length; i += info.channels) {
            const r = Math.floor(data[i] / (256 / bins));
            const g = Math.floor(data[i + 1] / (256 / bins));
            const b = Math.floor(data[i + 2] / (256 / bins));
            const idx = (r * bins * bins) + (g * bins) + b;
            histogram[idx]++;
        }

        return {
            bins,
            data: histogram,
            totalPixels: (data.length / info.channels)
        };
    }

    static async generateAdvancedEdgeSignature(buffer) {
        const edgeData = await sharp(buffer)
            .greyscale()
            .convolve({
                width: 3,
                height: 3,
                kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
            })
            .raw()
            .toBuffer();

        // Generate multi-scale edge signatures
        const signatures = await Promise.all([1.0, 0.75, 0.5].map(async scale => {
            const scaled = await sharp(buffer)
                .resize(
                    Math.round(scale * 1000),
                    Math.round(scale * 1000),
                    { fit: 'contain' }
                )
                .greyscale()
                .convolve({
                    width: 3,
                    height: 3,
                    kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
                })
                .normalize()
                .raw()
                .toBuffer();

            return {
                scale,
                signature: this.createEdgeHistogram(scaled)
            };
        }));

        return {
            multiScale: signatures,
            combined: signatures.map(s => s.signature).flat(),
            metadata: {
                scales: [1.0, 0.75, 0.5],
                blockSize: 16,
                timestamp: Date.now()
            }
        };
    }

    static async generateBinarySignature(buffer) {
        const data = await sharp(buffer)
            .greyscale()
            .resize(64, 64, { fit: 'fill' })
            .raw()
            .toBuffer();

        const threshold = this.calculateAdaptiveThreshold(data);
        return Buffer.from(data.map(pixel => pixel > threshold ? 1 : 0));
    }

    static async generateAdvancedHash(buffer, size) {
        const tempPath = path.join(os.tmpdir(), `${uuidv4()}.jpg`);
        try {
            await sharp(buffer)
                .normalize()
                .removeAlpha()
                .jpeg()
                .toBuffer()
                .then(normalized => fs.promises.writeFile(tempPath, normalized));

            const hash = await new Promise((resolve, reject) => {
                imageHash(tempPath, size, true, (error, hash) => {
                    if (error) reject(error);
                    else resolve(hash);
                });
            });

            const image = sharp(buffer);
            const { data, info } = await image
                .raw()
                .toBuffer({ resolveWithObject: true });

            const waveletHash = await this.generateWaveletHash(data, info.width, info.height);

            return {
                size,
                hash,
                wavelet: waveletHash,
                metadata: {
                    width: info.width,
                    height: info.height,
                    format: 'jpeg'
                }
            };

        } catch (error) {
            logger.error('Error generating advanced hash:', {
                error: error.message,
                stack: error.stack
            });
            throw error;
        } finally {
            try {
                if (fs.existsSync(tempPath)) {
                    fs.unlinkSync(tempPath);
                }
            } catch (cleanupError) {
                logger.warn('Error cleaning up temp file:', {
                    path: tempPath,
                    error: cleanupError.message
                });
            }
        }
    }

    static createEdgeHistogram(edgeData, bins = 32) {
        const histogram = new Array(bins).fill(0);
        for (const pixel of edgeData) {
            const bin = Math.min(bins - 1, Math.floor((pixel / 255) * bins));
            histogram[bin]++;
        }
        return histogram.map(count => count / edgeData.length);
    }

    static calculateAdaptiveThreshold(data) {
        const histogram = new Array(256).fill(0);
        for (const pixel of data) {
            histogram[pixel]++;
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
        const total = data.length;

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

static async generateWaveletHash(data, width, height) {
        const blockSize = 8;
        const result = [];
        
        for (let y = 0; y < height; y += blockSize) {
            for (let x = 0; x < width; x += blockSize) {
                let blockSum = 0;
                let count = 0;
                
                for (let by = 0; by < blockSize && (y + by) < height; by++) {
                    for (let bx = 0; bx < blockSize && (x + bx) < width; bx++) {
                        const idx = ((y + by) * width + (x + bx));
                        blockSum += data[idx];
                        count++;
                    }
                }
                
                result.push(blockSum / count);
            }
        }
        
        return result;
    }
}

// Core image processor implementation
const imageProcessor = {
    // File Operations
    getFileType(filename) {
        const ext = path.extname(filename).toLowerCase();
        return SUPPORTED_IMAGE_TYPES.includes(ext) ? 'image' : 'other';
    },

    isImage(filename) {
        return SUPPORTED_IMAGE_TYPES.includes(path.extname(filename).toLowerCase());
    },

    validateImageFile(file) {
        if (!file || !file.path) {
            logger.error('Invalid file provided');
            return false;
        }
        const isValid = this.isImage(file.path);
        logger.info('File validation result:', {
            path: file?.path,
            isValid,
            fileType: this.getFileType(file.path)
        });
        return isValid;
    },

    async generateImageHash(buffer, size = 16) {
        const tempPath = path.join(os.tmpdir(), `temp_${Date.now()}.png`);
        try {
            // Convert to PNG to ensure consistent format
            await sharp(buffer)
                .png()
                .toFile(tempPath);

            return new Promise((resolve, reject) => {
                imageHash(tempPath, size, true, (error, hash) => {
                    if (error) {
                        logger.warn('Image hash generation failed:', {
                            service: "tatt2awai-bot",
                            error: error.message,
                            fallback: true
                        });
                        // Fallback to basic buffer hash if image-hash fails
                        const fallbackHash = crypto.createHash('sha256')
                            .update(buffer)
                            .digest('hex')
                            .slice(0, size * 2);
                        resolve(fallbackHash);
                    } else {
                        resolve(hash);
                    }
                });
            });
        } finally {
            try {
                if (fs.existsSync(tempPath)) {
                    fs.unlinkSync(tempPath);
                }
            } catch (cleanupError) {
                logger.warn('Error cleaning up temp file:', {
                    service: "tatt2awai-bot",
                    error: cleanupError.message
                });
            }
        }
    },

    async generateFileHash(filePath) {
        const content = fs.readFileSync(filePath);
        return crypto.createHash('sha256').update(content).digest('hex');
    },

    async saveTempBuffer(buffer) {
        const tempPath = path.join(os.tmpdir(), `${uuidv4()}.jpg`);
        await fs.promises.writeFile(tempPath, buffer);
        return tempPath;
    },

    async calculateQuickScore(signature1, signature2) {
        try {
            if (!signature1 || !signature2) return 0;

            // Calculate aspect ratio similarity
            const aspectRatioScore = Math.min(
                signature1.metadata?.dimensions?.aspectRatio || 1,
                signature2.metadata?.dimensions?.aspectRatio || 1
            ) / Math.max(
                signature1.metadata?.dimensions?.aspectRatio || 1,
                signature2.metadata?.dimensions?.aspectRatio || 1
            );

            // Calculate color similarity
            const colorScore = this.compareColorMetrics(
                signature1.colorMetrics || {},
                signature2.colorMetrics || {}
            );

            // Calculate perceptual hash similarity
            const hashScore = await this.comparePerceptualHashes(
                signature1.perceptualHashes || [],
                signature2.perceptualHashes || []
            );

            // Weight and combine scores
            const weights = {
                aspectRatio: 0.2,
                color: 0.3,
                hash: 0.5
            };

            const finalScore = (
                aspectRatioScore * weights.aspectRatio +
                colorScore * weights.color +
                hashScore * weights.hash
            );

            return finalScore;
        } catch (error) {
            logger.error('Error calculating quick score:', {
                error: error.message,
                stack: error.stack
            });
            return 0;
        }
    },

compareColorMetrics(metrics1, metrics2) {
        if (!metrics1.means || !metrics2.means) return 0;

        // Compare color channel means
        const meanDiffs = metrics1.means.map((mean, i) => 
            Math.abs(mean - (metrics2.means[i] || 0)) / 255
        );

        // Calculate overall color similarity
        return 1 - (meanDiffs.reduce((a, b) => a + b, 0) / meanDiffs.length);
    },

    async comparePerceptualHashes(hashes1, hashes2) {
        if (!hashes1.length || !hashes2.length) return 0;

        // Compare hashes of the same size
        const scores = hashes1.map(hash1 => {
            const matchingHash = hashes2.find(h2 => h2.size === hash1.size);
            if (!matchingHash) return 0;
            return this.calculateHashSimilarity(hash1.hash, matchingHash.hash);
        });

        // Return best hash similarity score
        return Math.max(...scores);
    },

    // Core Image Processing
    async optimizeImage(filePath) {
        return await this.robustImageProcessing(async () => {
            try {
                const image = sharp(filePath);
                const metadata = await image.metadata();
                
                const processedImage = await sharp(filePath)
                    .resize({
                        width: Math.min(metadata.width, IMAGE_PROCESSING.MAX_SIZE),
                        height: Math.min(metadata.height, IMAGE_PROCESSING.MAX_SIZE),
                        fit: 'inside',
                        withoutEnlargement: true
                    })
                    .normalize()
                    .sharpen()
                    .jpeg({ quality: IMAGE_PROCESSING.QUALITY })
                    .toBuffer();

                return processedImage;
            } catch (error) {
                logger.error('Error optimizing image:', error);
                throw error;
            }
        });
    },

    async robustImageProcessing(buffer) {
        try {
            const strategies = [
                () => sharp(buffer).rotate(),
                () => sharp(buffer).normalize(),
                () => sharp(buffer).gamma(2.2)
            ];

            for (const strategy of strategies) {
                try {
                    return await strategy();
                } catch (e) {
                    continue;
                }
            }
            throw new Error('All processing strategies failed');
        } catch (error) {
            logger.error('Image processing failed:', error);
            throw error;
        }
    },

    // Main image processing method
async processImage(file) {
    let optimizedBuffer = null;
    let tempPath = null;
    let processingStage = 'validation';
    const startTime = Date.now();

    try {
        if (!file) {
            throw new Error('No file provided');
        }

        if (file.buffer) {
            tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path || 'temp.jpg')}`);
            fs.writeFileSync(tempPath, file.buffer);
            file.path = tempPath;
        }

        const buffer = await fs.promises.readFile(file.path);

        // Add validation with detailed error reporting
        const validation = await EnhancedValidation.validateImageContent(buffer);
        if (!validation.isValidImage) {
            throw new Error(`Image validation failed: ${validation.warnings.join(', ')}. Details: ${JSON.stringify(validation.details)}`);
        }      

        processingStage = 'optimization';
        const { optimizedBuffer: processedBuffer, metadata } = await ImageOptimizer.prepareImage(buffer);

        // Manage memory after optimization
        if (buffer !== processedBuffer) buffer = null;
        if (global.gc) global.gc();

        processingStage = 'signature';
        const signature = await EnhancedSignatureGenerator.generateSignature(processedBuffer);

        processingStage = 'storage';
        await EnhancedSignatureStore.set(file.path, signature);

        logger.info('Image processing completed', {
            path: file.path,
            processingTime: Date.now() - startTime,
            stages: ['validation', 'optimization', 'signature', 'storage']
        });

        return {
            signature,
            metadata,
            path: file.path,
            processingTime: Date.now() - startTime
        };

    } catch (error) {
        logger.error('Error in image processing:', {
            stage: processingStage,
            error: error.message,
            path: file?.path,
            processingTime: Date.now() - startTime,
            stack: error.stack
        });
        throw error;
    } finally {
        // Cleanup
        if (tempPath && fs.existsSync(tempPath)) {
            try {
                fs.unlinkSync(tempPath);
            } catch (cleanupError) {
                logger.warn('Failed to cleanup temp file:', {
                    path: tempPath,
                    error: cleanupError.message
                });
            }
        }
        // Clear buffers
        optimizedBuffer = null;
        if (global.gc) global.gc();
    }
},

    async prefilterCandidates(allImages, searchMetadata) {
        // Don't filter by aspect ratio initially - use it as a scoring factor instead
        return allImages.filter(image => 
            image.path_lower.match(/\.(jpg|jpeg|png|webp)$/i)
        );
    },

    async extractImageFeatures(buffer) {
        const image = sharp(buffer);
        
        // Get multiple resolution hashes and features in parallel
        const [
            metadata,
            colorHistogram,
            smallHash,
            mediumHash,
            largeHash,
            edges
        ] = await Promise.all([
            image.metadata(),
            this.generateColorHistogram(buffer),
            this.generateHash(buffer, 8),   // Small hash for quick comparison
            this.generateHash(buffer, 32),  // Medium hash for better accuracy
            this.generateHash(buffer, 64),  // Large hash for final verification
            this.generateEdgeSignature(buffer)
        ]);

        return {
            metadata,
            colorHistogram,
            hashes: { small: smallHash, medium: mediumHash, large: largeHash },
            edges,
            aspectRatio: metadata.width / metadata.height
        };
    },

async generateColorHistogram(buffer) {
        const data = await sharp(buffer)
            .resize(64, 64, { fit: 'fill' })
            .raw()
            .toBuffer();

        const histogram = new Array(64).fill(0);
        for (let i = 0; i < data.length; i += 3) {
            const r = data[i], g = data[i + 1], b = data[i + 2];
            const index = Math.floor((r + g + b) / 12); // Simplified color binning
            histogram[index]++;
        }

        return histogram;
    },

    async generateEdgeSignature(buffer) {
        return sharp(buffer)
            .resize(32, 32)
            .greyscale()
            .convolve({
                width: 3,
                height: 3,
                kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
            })
            .raw()
            .toBuffer();
    },

downloadQueue: new DownloadQueue(SEARCH_CONSTANTS.DOWNLOAD.MAX_CONCURRENT),

    async downloadWithRetry(path, dropboxManager) {
        for (let attempt = 0; attempt <= SEARCH_CONSTANTS.DOWNLOAD.MAX_RETRIES; attempt++) {
            try {
                const downloadPromise = dropboxManager.downloadFile(path);
                const timeoutPromise = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Download timeout')), 
                    SEARCH_CONSTANTS.DOWNLOAD.TIMEOUT)
                );

                const fileData = await this.downloadQueue.add(async () => 
                    Promise.race([downloadPromise, timeoutPromise])
                );

                if (fileData?.result?.fileBinary) {
                    return Buffer.isBuffer(fileData.result.fileBinary) ?
                        fileData.result.fileBinary :
                        Buffer.from(fileData.result.fileBinary);
                }
                return null;
            } catch (error) {
                if (attempt === SEARCH_CONSTANTS.DOWNLOAD.MAX_RETRIES) {
                    logger.warn(`Download failed for ${path} after ${attempt + 1} attempts:`, {
                        service: "tatt2awai-bot",
                        error: error.message
                    });
                    return null;
                }
                // Wait before retrying with exponential backoff
                await new Promise(resolve => 
                    setTimeout(resolve, SEARCH_CONSTANTS.DOWNLOAD.RETRY_DELAYS[attempt])
                );
            }
        }
        return null;
    },

    async processWithTimeout(promise, timeoutMs, operation) {
        try {
            return await Promise.race([
                promise,
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error(`Timeout: ${operation}`)), timeoutMs)
                )
            ]);
        } catch (error) {
            if (error.message.startsWith('Timeout:')) {
                logger.warn(`Operation timed out: ${operation}`);
            }
            throw error;
        }
    },

    withTimeout: async function(promise, timeoutMs, operation) {
        return Promise.race([
            promise,
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error(`Timeout: ${operation}`)), timeoutMs)
            )
        ]);
    },

    processInParallel: async function(items, processor, maxParallel = 5) {
        const results = [];
        const chunks = [];
        
        for (let i = 0; i < items.length; i += SEARCH_CONSTANTS.BATCH_SIZE) {
            chunks.push(items.slice(i, i + SEARCH_CONSTANTS.BATCH_SIZE));
        }

        for (let i = 0; i < chunks.length; i += maxParallel) {
            const batch = chunks.slice(i, i + maxParallel);
            const batchResults = await Promise.all(
                batch.map(chunk => processor(chunk))
            );
            results.push(...batchResults.flat());

            if (results.some(r => r?.score > 0.98)) {
                break;
            }
        }

        return results;
    },

processChunk: async function(chunk, searchBuffer, searchHash, dropboxManager, metrics) {
    const chunkResults = await Promise.all(chunk.map(async (image) => {
        try {
            // Download with retry
            const fileData = await this.retryDownload(image.path_lower, dropboxManager);
            if (!fileData) return null;

            metrics.downloads++;
            const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                fileData.result.fileBinary :
                Buffer.from(fileData.result.fileBinary);

            // Multi-stage comparison
            const similarityScore = await this.compareImage(searchBuffer, buffer);

            if (similarityScore > SEARCH_CONSTANTS.THRESHOLDS.QUICK_MATCH) {
                logger.info('Match found:', {
                    service: "tatt2awai-bot",
                    path: image.path_lower,
                    score: similarityScore
                });

                return {
                    path: image.path_lower,
                    buffer,
                    score: similarityScore
                };
            }
            return null;
        } catch (error) {
            return null;
        } finally {
            metrics.processed++;
        }
    }));

    return chunkResults.filter(Boolean);
},

async findExactMatch(searchBuffer, allImages, dropboxManager) {
    try {
        // Generate signature for search image
        const searchSignature = await EnhancedSignatureGenerator.generateSignature(searchBuffer);
        let bestMatch = null;
        let bestScore = 0;

        // Quick filtering by aspect ratio first
        const potentialMatches = allImages.result.entries.filter(image => {
            const signature = EnhancedSignatureStore.get(image.path_lower);
            if (!signature) return false;
            return Math.abs(signature.metadata.aspectRatio - searchSignature.metadata.aspectRatio) < 0.1;
        });

        // Process potential matches
        for (const image of potentialMatches) {
            try {
                const fileData = await dropboxManager.downloadFile(image.path_lower);
                if (!fileData?.result?.fileBinary) continue;

                const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                    fileData.result.fileBinary :
                    Buffer.from(fileData.result.fileBinary);

                const comparison = await this.compareImage(searchBuffer, buffer);
                
                if (comparison.score > 0.95) {
                    return {
                        path: image.path_lower,
                        similarity: comparison.score,
                        confidence: comparison.confidence,
                        matchType: 'exact'
                    };
                }

                if (comparison.score > bestScore) {
                    bestScore = comparison.score;
                    bestMatch = image;
                }
            } catch (error) {
                logger.warn(`Failed to process ${image.path_lower}:`, error);
                continue;
            }
        }

        return bestMatch ? {
            path: bestMatch.path_lower,
            similarity: bestScore,
            confidence: bestScore > 0.9 ? 'high' : 'medium',
            matchType: 'similar'
        } : null;

    } catch (error) {
        logger.error('Search error:', error);
        throw error;
    }
},


quickCompare: async function(searchBuffer, candidateBuffer) {
    try {
        const [searchThumb, candidateThumb] = await Promise.all([
            sharp(searchBuffer)
                .resize(32, 32, { fit: 'contain' })
                .removeAlpha()
                .raw()
                .toBuffer(),
            sharp(candidateBuffer)
                .resize(32, 32, { fit: 'contain' })
                .removeAlpha()
                .raw()
                .toBuffer()
        ]);

        return this.compareBuffers(searchThumb, candidateThumb);
    } catch (error) {
        return 0;
    }
},


quickVerify: async function(searchBuffer, matchBuffer) {
    try {
        const [search, match] = await Promise.all([
            sharp(searchBuffer)
                .resize(128, 128, { fit: 'contain' })
                .raw()
                .toBuffer(),
            sharp(matchBuffer)
                .resize(128, 128, { fit: 'contain' })
                .raw()
                .toBuffer()
        ]);

        const pixelScore = this.compareBuffers(search, match);
        
        // Log verification scores
        logger.info('Verification scores:', {
            service: "tatt2awai-bot",
            pixelScore,
            threshold: SEARCH_CONSTANTS.THRESHOLDS.GOOD_MATCH
        });

        return pixelScore;
    } catch (error) {
        return 0;
    }
},

compareBuffers: function(buf1, buf2) {
    if (buf1.length !== buf2.length) return 0;
    
    let diff = 0;
    const pixelCount = buf1.length;
    
    // Compare each byte with some tolerance
    for (let i = 0; i < pixelCount; i++) {
        const delta = Math.abs(buf1[i] - buf2[i]);
        // Allow some tolerance in pixel differences
        diff += delta < 10 ? 0 : delta;
    }
    
    return 1 - (diff / (pixelCount * 255));
},

    compareHashes: function(hash1, hash2) {
        if (hash1.length !== hash2.length) return 0;
        const differences = hash1.reduce((count, bit, i) => 
            count + (bit === hash2[i] ? 0 : 1), 0);
        return 1 - (differences / hash1.length);
    },


async compareImage(searchBuffer, candidateBuffer) {
    const [searchSignature, candidateSignature] = await Promise.all([
        EnhancedSignatureGenerator.generateSignature(searchBuffer),
        EnhancedSignatureGenerator.generateSignature(candidateBuffer)
    ]);
    
    const comparison = EnhancedSignatureStore.compareSignatures(searchSignature, candidateSignature);
    
    return {
        score: comparison.score,
        confidence: comparison.confidence,
        components: comparison.components
    };
},

    async compareColorHistograms(buf1, buf2) {
        const [hist1, hist2] = await Promise.all([
            this.generateColorHistogram(buf1),
            this.generateColorHistogram(buf2)
        ]);

        return this.compareHistograms(hist1, hist2);
    },

    async generateColorHistogram(buffer) {
        const data = await sharp(buffer)
            .resize(64, 64, { fit: 'contain' })
            .raw()
            .toBuffer();

        const histogram = new Array(64).fill(0);
        for (let i = 0; i < data.length; i += 3) {
            const r = data[i], g = data[i + 1], b = data[i + 2];
            const bin = Math.floor((r + g + b) / 12);
            histogram[bin]++;
        }
        return histogram;
    },

    compareHistograms(hist1, hist2) {
        const sum1 = hist1.reduce((a, b) => a + b, 0);
        const sum2 = hist2.reduce((a, b) => a + b, 0);
        
        let similarity = 0;
        for (let i = 0; i < hist1.length; i++) {
            const h1 = hist1[i] / sum1;
            const h2 = hist2[i] / sum2;
            similarity += Math.min(h1, h2);
        }
        return similarity;
    },

async comparePixelContent(buf1, buf2) {
    if (buf1.length !== buf2.length) return 0;
    
    let diff = 0;
    for (let i = 0; i < buf1.length; i++) {
        const delta = Math.abs(buf1[i] - buf2[i]);
        diff += delta < 10 ? 0 : delta;  // Allow small differences
    }
    
    return 1 - (diff / (buf1.length * 255));
},

async compareEdgeContent(buf1, buf2) {
    const edges1 = await this.detectEdges(buf1);
    const edges2 = await this.detectEdges(buf2);
    return this.comparePixelContent(edges1, edges2);
},


    // Update the process chunk function to use the new comparison
    async processChunk(chunk, searchBuffer, searchHash, dropboxManager, metrics) {
        const chunkResults = await Promise.all(chunk.map(async (image) => {
            try {
                // Download with retry
                const fileData = await this.retryDownload(image.path_lower, dropboxManager);
                if (!fileData) return null;

                metrics.downloads++;
                const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                    fileData.result.fileBinary :
                    Buffer.from(fileData.result.fileBinary);

                // Multi-stage comparison
                const similarityScore = await this.compareImage(searchBuffer, buffer);

                if (similarityScore > SEARCH_CONSTANTS.THRESHOLDS.QUICK_MATCH) {
                    logger.info('Match found:', {
                        service: "tatt2awai-bot",
                        path: image.path_lower,
                        score: similarityScore
                    });

                    return {
                        path: image.path_lower,
                        buffer,
                        score: similarityScore
                    };
                }
                return null;
            } catch (error) {
                return null;
            } finally {
                metrics.processed++;
            }
        }));

        return chunkResults.filter(Boolean);
    },

    // Add retry logic for downloads
    async retryDownload(path, dropboxManager, maxRetries = 2) {
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const fileData = await this.withTimeout(
                    dropboxManager.downloadFile(path),
                    SEARCH_CONSTANTS.TIMEOUTS.DOWNLOAD,
                    'Download'
                );
                return fileData;
            } catch (error) {
                if (attempt === maxRetries) {
                    logger.warn(`Download failed after ${maxRetries} retries:`, {
                        path,
                        error: error.message
                    });
                    return null;
                }
                await new Promise(resolve => 
                    setTimeout(resolve, SEARCH_CONSTANTS.TIMEOUTS.RETRY_DELAY * (attempt + 1))
                );
            }
        }
        return null;
    },

    prefilterCandidates(images, searchMetadata) {
        return images.filter(image => {
            if (!image.metadata?.width || !image.metadata?.height) return false;
            
            const aspectRatio = image.metadata.width / image.metadata.height;
            const searchAspectRatio = searchMetadata.width / searchMetadata.height;
            
            return Math.abs(aspectRatio - searchAspectRatio) < 0.1;
        });
    },

    async downloadAndProcess(path, dropboxManager) {
        try {
            return await this.downloadWithRetry(path, dropboxManager);
        } catch (error) {
            logger.warn(`Failed to download/process ${path} after retries:`, {
                service: "tatt2awai-bot",
                error: error.message
            });
            return null;
        }
    },

compareDetailed: async function(searchBuffer, image, dropboxManager) {
    const [searchThumb, imageThumb] = await Promise.all([
        this.generateDetailedThumb(searchBuffer),
        this.getImageDetailedThumb(image, dropboxManager)
    ]);

    if (!imageThumb) return 0;

    const pixelScore = await this.comparePixelContent(searchThumb, imageThumb);
    const edgeScore = await this.compareEdgeContent(searchThumb, imageThumb);
    
    return (pixelScore * 0.6) + (edgeScore * 0.4);
},

async generateDetailedThumb(buffer) {
    return sharp(buffer)
        .resize(256, 256, { fit: 'contain' })
        .normalize()
        .removeAlpha()
        .raw()
        .toBuffer();
},

async getImageDetailedThumb(image, dropboxManager) {
    try {
        // Check cache first
        const cached = await this.getCachedThumb(image.path_lower);
        if (cached) return cached;

        // Download and process
        const data = await dropboxManager.downloadFile(image.path_lower);
        if (!data?.result?.fileBinary) return null;

        const buffer = Buffer.isBuffer(data.result.fileBinary) ?
            data.result.fileBinary :
            Buffer.from(data.result.fileBinary);

        const thumb = await this.generateDetailedThumb(buffer);
        await this.cacheThumb(image.path_lower, thumb);
        
        return thumb;
    } catch (error) {
        logger.warn(`Failed to get detailed thumb for ${image.path_lower}:`, error);
        return null;
    }
},

async generateStandardThumbnail(buffer) {
        return sharp(buffer)
            .resize(256, 256, { fit: 'contain' })
            .normalize()
            .removeAlpha()
            .raw()
            .toBuffer();
    },

    async promiseWithTimeout(promise, timeoutMs, operationName) {
        try {
            return await Promise.race([
                promise,
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error(`${operationName} timeout`)), timeoutMs)
                )
            ]);
        } catch (error) {
            if (error.message.includes('timeout')) {
                logger.warn(`Operation timed out: ${operationName}`, {
                    service: "tatt2awai-bot",
                    timeoutMs
                });
            }
            throw error;
        }
    },

    async generateThumbnail(buffer, size) {
        return sharp(buffer)
            .resize(size, size, {
                fit: 'contain',
                background: { r: 255, g: 255, b: 255, alpha: 1 }
            })
            .normalize()
            .grayscale()
            .raw()
            .toBuffer();
    },

async generateBasicImageMetadata(buffer) {

        const metadata = await sharp(buffer).metadata();
        return {
            aspectRatio: metadata.width / metadata.height,
            size: metadata.size,
            format: metadata.format
        };
    },

    filterImageByMetadata(image, searchMetadata) {
        if (!image.path_lower.match(/\.(jpg|jpeg|png|webp)$/i)) return false;
        
        if (image.metadata?.width && image.metadata?.height) {
            const imageAspectRatio = image.metadata.width / image.metadata.height;
            const searchAspectRatio = searchMetadata.width / searchMetadata.height;
            return Math.abs(imageAspectRatio - searchAspectRatio) < MATCH_CONSTANTS.THRESHOLDS.ASPECT_RATIO;
        }

        return true;
    },

    async compareQuick(thumb1, thumb2) {
        if (thumb1.length !== thumb2.length) {
            return 0;
        }

        let diff = 0;
        for (let i = 0; i < thumb1.length; i++) {
            diff += Math.abs(thumb1[i] - thumb2[i]);
        }
        return 1 - (diff / (thumb1.length * 255));
    },

    async processMatch(searchBuffer, candidateBuffer, path, quickScore) {
        try {
            // Generate detailed thumbnails for better comparison
            const [detailedSearch, detailedCandidate] = await Promise.all([
                this.generateThumbnail(searchBuffer, MATCH_CONSTANTS.DIMENSIONS.DETAILED),
                this.generateThumbnail(candidateBuffer, MATCH_CONSTANTS.DIMENSIONS.DETAILED)
            ]);

            // Detailed comparison
            const detailedScore = await this.compareQuick(detailedSearch, detailedCandidate);
            
            if (detailedScore > MATCH_CONSTANTS.THRESHOLDS.DETAILED_MATCH) {
                // Final high-resolution comparison for promising matches
                const [finalSearch, finalCandidate] = await Promise.all([
                    this.generateThumbnail(searchBuffer, MATCH_CONSTANTS.DIMENSIONS.FINAL),
                    this.generateThumbnail(candidateBuffer, MATCH_CONSTANTS.DIMENSIONS.FINAL)
                ]);

                const finalScore = await this.compareQuick(finalSearch, finalCandidate);

                if (finalScore > MATCH_CONSTANTS.THRESHOLDS.DETAILED_MATCH) {
                    return {
                        path,
                        similarity: finalScore,
                        confidence: (quickScore + detailedScore + finalScore) / 3,
                        matchType: finalScore > MATCH_CONSTANTS.THRESHOLDS.PERFECT_MATCH ? 'exact' : 'high',
                        scores: {
                            quick: quickScore,
                            detailed: detailedScore,
                            final: finalScore
                        }
                    };
                }
            }
            return null;
        } catch (error) {
            logger.warn(`Error in detailed comparison for ${path}:`, {
                service: "tatt2awai-bot",
                error: error.message
            });
            return null;
        }
    },

    async verifyAndRankMatches(matches, searchBuffer) {
        if (!matches.length) return null;

        // Sort by similarity first
        const sortedMatches = matches.sort((a, b) => b.similarity - a.similarity);
        
        // Take top 3 matches for final verification
        const topMatches = sortedMatches.slice(0, 3);

        try {
            const verifiedMatches = await Promise.all(
                topMatches.map(async match => {
                    try {
                        // Final verification at maximum resolution
                        const verificationScore = await verifyMatch(searchBuffer, match);
                        return {
                            ...match,
                            verificationScore,
                            finalScore: (match.similarity + verificationScore) / 2
                        };
                    } catch (error) {
                        logger.warn(`Verification failed for ${match.path}:`, {
                            service: "tatt2awai-bot",
                            error: error.message
                        });
                        return match;
                    }
                })
            );

            // Return the best verified match
            return verifiedMatches
                .filter(m => m.verificationScore > MATCH_CONSTANTS.THRESHOLDS.DETAILED_MATCH)
                .sort((a, b) => b.finalScore - a.finalScore)[0] || sortedMatches[0];

        } catch (error) {
            logger.error('Error in match verification:', {
                service: "tatt2awai-bot",
                error: error.message
            });
            return sortedMatches[0];
        }
    },

async verifyMatch(searchBuffer, image, dropboxManager) {
    const data = await dropboxManager.downloadFile(image.path_lower);
    if (!data?.result?.fileBinary) return 0;

    const buffer = Buffer.isBuffer(data.result.fileBinary) ?
        data.result.fileBinary :
        Buffer.from(data.result.fileBinary);

    const [verifySearch, verifyImage] = await Promise.all([
        this.generateVerificationThumb(searchBuffer),
        this.generateVerificationThumb(buffer)
    ]);

    return this.comparePixelContent(verifySearch, verifyImage);
},

async generateVerificationThumb(buffer) {
    return sharp(buffer)
        .resize(512, 512, { fit: 'contain' })
        .normalize()
        .removeAlpha()
        .raw()
        .toBuffer();
},

    async generatePerceptualHash(buffer) {
        const hashSizes = [8, 16, 32];
        const hashes = await Promise.all(hashSizes.map(async size => {
            const resized = await sharp(buffer)
                .resize(size, size, { fit: 'fill' })
                .grayscale()
                .raw()
                .toBuffer();

            return {
                size,
                hash: crypto.createHash('sha256').update(resized).digest('hex')
            };
        }));

        return hashes;
    },

    async manageMemory(tempCache) {
        if (process.memoryUsage().heapUsed > 500 * 1024 * 1024) { // 500MB
            tempCache.clear();
            if (global.gc) global.gc();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    },

    compareQuickSignatures(sig1, sig2) {
        const aspectRatioMatch = Math.abs(sig1.aspectRatio - sig2.aspectRatio) < 
                            MATCH_CONSTANTS.ASPECT_RATIO_TOLERANCE;
        if (!aspectRatioMatch) return 0;

        const colorDiff = sig1.averageColor.reduce((diff, value, i) => 
            diff + Math.abs(value - sig2.averageColor[i]), 0) / sig1.averageColor.length;

        return 1 - (colorDiff / 255);
    },

    async processCandidate(image, searchBuffer, searchHash, dropboxManager) {
        const fileData = await dropboxManager.downloadFile(image.path_lower);
        if (!fileData?.result?.fileBinary) return null;

        const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
            fileData.result.fileBinary :
            Buffer.from(fileData.result.fileBinary);

        const imageHash = await ImageHasher.generateHash(buffer);
        const score = ImageHasher.calculateSimilarity(searchHash, imageHash);

        if (score <= SEARCH_CONSTANTS.MIN_SCORE) return null;

        return {
            path: image.path_lower,
            buffer,
            score
        };
    },

    async processBatchWithLimit(batch, searchBuffer, cache, concurrentLimit) {
        const results = [];
        for (let i = 0; i < batch.length; i += concurrentLimit) {
            const chunk = batch.slice(i, i + concurrentLimit);
            const chunkResults = await Promise.all(
                chunk.map(image => processCandidate(searchBuffer, image, cache))
            );
            results.push(...chunkResults.filter(Boolean));
            await new Promise(resolve => setTimeout(resolve, 100)); // Rate limiting
        }
        return results;
    },

    async generateQuickSignature(buffer) {
        const stats = await sharp(buffer).stats();
        const metadata = await sharp(buffer).metadata();
        
        return {
            aspectRatio: metadata.width / metadata.height,
            averageColor: stats.channels.map(c => c.mean),
            size: metadata.size,
            format: metadata.format
        };
    },

prefilterBatch(batch, searchSignature) {
        return batch.filter(image => {
            // Quick metadata-based filtering
            if (path.extname(image.path_lower) !== '.jpg' && 
                path.extname(image.path_lower) !== '.jpeg' && 
                path.extname(image.path_lower) !== '.png') {
                return false;
            }
            
            return true;
        });
    },

    async downloadAndNormalize(image, dropboxManager) {
        const fileData = await dropboxManager.downloadFile(image.path_lower);
        if (!fileData?.result?.fileBinary) return null;

        const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
            fileData.result.fileBinary :
            Buffer.from(fileData.result.fileBinary);

        const normalized = await sharp(buffer)
            .resize(512, 512, {
                fit: 'inside',
                withoutEnlargement: true
            })
            .normalize()
            .removeAlpha()
            .toBuffer();

        return {
            original: buffer,
            normalized
        };
    },

    async processCandidate(imageData, searchSignature, normalizedSearchBuffer, image, metrics) {
        // Quick signature comparison
        const candidateSignature = await this.generateQuickSignature(imageData.normalized);
        const quickScore = this.compareQuickSignatures(searchSignature, candidateSignature);

        if (quickScore < MIN_QUICK_MATCH) {
            return null;
        }

        metrics.comparisons++;

        // Detailed pixel comparison for promising candidates
        const similarity = await this.compareBuffers(normalizedSearchBuffer, imageData.normalized);

        if (similarity > EXACT_MATCH_THRESHOLD) {
            const verificationScore = await this.verifyMatch(
                normalizedSearchBuffer,
                imageData.normalized
            );

            if (verificationScore > EXACT_MATCH_THRESHOLD) {
                return {
                    path: image.path_lower,
                    similarity: verificationScore,
                    matchType: 'exact',
                    confidence: verificationScore
                };
            }
        }

        return null;
    },

    compareQuickSignatures(sig1, sig2) {
        const aspectRatioMatch = Math.abs(sig1.aspectRatio - sig2.aspectRatio) < 0.1;
        if (!aspectRatioMatch) return 0;

        const colorDiff = sig1.averageColor.reduce((diff, value, i) => 
            diff + Math.abs(value - sig2.averageColor[i]), 0) / sig1.averageColor.length;

        return 1 - (colorDiff / 255);
    },

    async compareBuffers(buffer1, buffer2) {
        const diff = await this.calculatePixelDifference(buffer1, buffer2);
        return 1 - (diff / 255);
    },

    async calculatePixelDifference(buffer1, buffer2) {
        const data1 = new Uint8Array(buffer1);
        const data2 = new Uint8Array(buffer2);
        let totalDiff = 0;
        const length = Math.min(data1.length, data2.length);

        for (let i = 0; i < length; i++) {
            totalDiff += Math.abs(data1[i] - data2[i]);
        }

        return totalDiff / length;
    },

async detectEdges(buffer) {
    return sharp(buffer)
        .convolve({
            width: 3,
            height: 3,
            kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
        })
        .raw()
        .toBuffer();
},

    compareEdgePatterns(edges1, edges2) {
        const data1 = new Uint8Array(edges1);
        const data2 = new Uint8Array(edges2);
        let matches = 0;
        const length = Math.min(data1.length, data2.length);

        for (let i = 0; i < length; i++) {
            if (Math.abs(data1[i] - data2[i]) < 30) {
                matches++;
            }
        }

        return matches / length;
    },

    async validateMatch(buffer1, buffer2) {
        const tempFiles = [];
        try {
            const temp1 = path.join(os.tmpdir(), `validate1_${Date.now()}.jpg`);
            const temp2 = path.join(os.tmpdir(), `validate2_${Date.now()}.jpg`);
            tempFiles.push(temp1, temp2);

            await Promise.all([
                fs.promises.writeFile(temp1, buffer1),
                fs.promises.writeFile(temp2, buffer2)
            ]);

            const [processed1, processed2] = await Promise.all([
                sharp(temp1)
                    .resize(1000, 1000, { fit: 'inside', withoutEnlargement: true })
                    .raw()
                    .toBuffer({ resolveWithObject: true }),
                sharp(temp2)
                    .resize(1000, 1000, { fit: 'inside', withoutEnlargement: true })
                    .raw()
                    .toBuffer({ resolveWithObject: true })
            ]);

            const result = ssim(
                { data: processed1.data, width: 1000, height: 1000, channels: 3 },
                { data: processed2.data, width: 1000, height: 1000, channels: 3 }
            );

            return result.mssim > 0.95;

        } catch (error) {
            logger.error('Error validating match:', {
                error: error.message,
                stack: error.stack
            });
            return false;
        } finally {
            // Clean up temp files
            for (const tempFile of tempFiles) {
                try {
                    if (fs.existsSync(tempFile)) {
                        await fs.promises.unlink(tempFile);
                    }
                } catch (cleanupError) {
                    logger.warn(`Error cleaning up temp file ${tempFile}:`, {
                        error: cleanupError.message
                    });
                }
            }
        }
    },

async findImageSequence(matchedImage, allImages) {
    const directory = path.dirname(matchedImage.path_lower);
    const dirImages = allImages.result.entries
        .filter(img => path.dirname(img.path_lower) === directory)
        .sort((a, b) => {
            // First try sequence numbers in filename
            const seqA = this.extractSequenceNumber(a.name);
            const seqB = this.extractSequenceNumber(b.name);
            if (seqA !== null && seqB !== null) return seqA - seqB;

            // Fall back to timestamps
            return new Date(a.server_modified) - new Date(b.server_modified);
        });

    const matchIndex = dirImages.findIndex(img => 
        img.path_lower === matchedImage.path_lower
    );

    if (matchIndex === -1) return null;

    return {
        sequence: dirImages,
        current: matchIndex,
        before: dirImages.slice(0, matchIndex),
        after: dirImages.slice(matchIndex + 1)
    };
},

    extractSequenceNumber(filename) {
        // Match common sequence patterns in filenames
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
                // Handle before/after
                if (match[0].toLowerCase() === 'before') return 0;
                if (match[0].toLowerCase() === 'after') return 9999;
            }
        }
        return null;
    },

    async generateHash(buffer, size) {
        const data = await sharp(buffer)
            .resize(size, size, { fit: 'fill' })
            .greyscale()
            .raw()
            .toBuffer();

        const pixels = new Uint8Array(data);
        const mean = pixels.reduce((sum, val) => sum + val, 0) / pixels.length;
        return pixels.map(p => p > mean ? 1 : 0);
    },

    async analyzeTattooFeatures(visionResult) {
        try {
            if (!visionResult) return null;

            // Extract relevant annotations
            const labels = visionResult.labelAnnotations || [];
            const properties = visionResult.imagePropertiesAnnotation || {};
            const objects = visionResult.localizedObjectAnnotations || [];

            // Basic feature detection
            const isTattooRelated = this.detectTattooContent(labels);
            const inkPatterns = await this.analyzeInkPatterns(properties);
            const skinFeatures = this.analyzeSkinFeatures(properties, objects);
            const edgeDefinition = await this.analyzeEdgeDefinition(properties);

            // Analyze potential removal indicators
            const removalIndicators = this.analyzeRemovalIndicators(labels, properties);

            return {
                isTattoo: isTattooRelated.isTattoo,
                confidence: isTattooRelated.confidence,
                features: {
                    ink: inkPatterns,
                    skin: skinFeatures,
                    edges: edgeDefinition
                },
                removal: removalIndicators,
                metadata: {
                    analyzedAt: new Date().toISOString(),
                    version: '2.0'
                }
            };
        } catch (error) {
            logger.error('Error analyzing tattoo features:', {
                error: error.message,
                stack: error.stack
            });
            return null;
        }
    },

    detectTattooContent(labels) {
        const tattooRelatedLabels = labels.filter(label => 
            TATTOO_KEYWORDS.has(label.description.toLowerCase())
        );

        const confidence = tattooRelatedLabels.length > 0 
            ? Math.max(...tattooRelatedLabels.map(l => l.score))
            : 0;

        return {
            isTattoo: confidence > ANALYSIS_THRESHOLDS.TATTOO_CONFIDENCE,
            confidence,
            matchedLabels: tattooRelatedLabels.map(l => l.description)
        };
    },

async analyzeInkPatterns(properties) {
        const colors = properties.dominantColors?.colors || [];
        const darkColors = colors.filter(color => {
            const brightness = (color.color.red + color.color.green + color.color.blue) / (3 * 255);
            return brightness < 0.4 && color.pixelFraction > 0.02;
        });

        return {
            inkDensity: this.calculateInkDensity(properties),
            patterns: this.detectPatterns(darkColors),
            colorProfile: this.analyzeColorProfile(colors)
        };
    },

    analyzeSkinFeatures(properties, objects) {
        const skinObjects = objects.filter(obj => 
            obj.name.toLowerCase().includes('skin') ||
            obj.name.toLowerCase().includes('person')
        );

        const skinTones = this.detectSkinTones(properties.dominantColors?.colors || []);

        return {
            hasSkin: skinObjects.length > 0,
            confidence: skinObjects[0]?.score || 0,
            tones: skinTones,
            coverage: this.calculateSkinCoverage(skinObjects)
        };
    },

    async analyzeEdgeDefinition(properties) {
        const colors = properties.dominantColors?.colors || [];
        const edges = await this.detectEdges(colors);

        return {
            sharpness: edges.sharpness || 0,
            clarity: edges.clarity || 0,
            pattern: edges.pattern || 'unknown'
        };
    },

    analyzeRemovalIndicators(labels, properties) {
        const removalKeywords = ['fading', 'removal', 'treatment', 'laser'];
        const hasRemovalLabels = labels.some(label => 
            removalKeywords.some(keyword => 
                label.description.toLowerCase().includes(keyword)
            )
        );

        const inkDensity = this.calculateInkDensity(properties);
        const fragmentationLevel = this.detectFragmentationLevel(properties);

        return {
            isRemovalInProgress: hasRemovalLabels || inkDensity < ANALYSIS_THRESHOLDS.INK_DENSITY.MEDIUM,
            stage: this.determineRemovalStage(inkDensity, fragmentationLevel),
            confidence: Math.max(
                ...labels.filter(l => removalKeywords.some(k => 
                    l.description.toLowerCase().includes(k)
                )).map(l => l.score),
                0
            )
        };
    },

    detectEdges(colors) {
        return {
            sharpness: 0.8,
            clarity: 0.8,
            pattern: 'defined'
        };
    },

    detectSkinTones(colors) {
        return colors
            .filter(color => this.isSkinTone(color.color))
            .map(color => ({
                rgb: color.color,
                prominence: color.pixelFraction
            }));
    },

    isSkinTone(color) {
        return ENHANCED_PATTERNS.SKIN_TONES.some(tone => 
            color.red >= tone.min[0] && color.red <= tone.max[0] &&
            color.green >= tone.min[1] && color.green <= tone.max[1] &&
            color.blue >= tone.min[2] && color.blue <= tone.max[2]
        );
    },

    calculateSkinCoverage(skinObjects) {
        if (!skinObjects.length) return 0;
        return skinObjects.reduce((total, obj) => total + this.calculateBoundingBoxArea(obj), 0);
    },

    calculateBoundingBoxArea(object) {
        if (!object.boundingPoly?.normalizedVertices) return 0;
        const vertices = object.boundingPoly.normalizedVertices;
        return (vertices[1]?.x - vertices[0]?.x) * (vertices[2]?.y - vertices[1]?.y) || 0;
    },

    detectPatterns(darkColors) {
        return {
            isDotted: darkColors.length > ANALYSIS_THRESHOLDS.PATTERN_DETECTION.MINIMUM_DOTS,
            density: darkColors.reduce((sum, color) => sum + color.pixelFraction, 0),
            confidence: darkColors.length > 0 ? 
                Math.min(darkColors.reduce((sum, color) => sum + color.pixelFraction, 0), 1) : 0
        };
    },

    analyzeColorProfile(colors) {
        return colors.map(color => ({
            rgb: color.color,
            prominence: color.pixelFraction,
            isInk: this.isInkLikeColor(color.color)
        }));
    },

    isInkLikeColor(color) {
        const brightness = (color.red + color.green + color.blue) / (3 * 255);
        return brightness < 0.4;
    },
detectFragmentationLevel(properties) {
        const colors = properties.dominantColors?.colors || [];
        const darkColors = colors.filter(color => this.isInkLikeColor(color.color));
        
        if (darkColors.length === 0) return 0;
        
        const totalCoverage = darkColors.reduce((sum, color) => sum + color.pixelFraction, 0);
        const averageCoverage = totalCoverage / darkColors.length;
        
        return Math.min(averageCoverage * 3, 1); // Normalize to 0-1 range
    },

    determineRemovalStage(inkDensity, fragmentationLevel) {
        if (inkDensity < ANALYSIS_THRESHOLDS.INK_DENSITY.MINIMAL) return 'near_complete';
        if (inkDensity < ANALYSIS_THRESHOLDS.INK_DENSITY.LOW) return 'significant_fading';
        if (inkDensity < ANALYSIS_THRESHOLDS.INK_DENSITY.MEDIUM) return 'moderate_fading';
        return 'early_stage';
    },

    // Initialize comparison libraries
    async initializeComparisonLibraries() {
        try {
            const ssimModule = require('ssim.js');
            
            logger.info('Detailed SSIM module inspection:', {
                service: "tatt2awai-bot",
                moduleType: typeof ssimModule,
                moduleKeys: Object.keys(ssimModule),
                ssimType: typeof ssimModule.ssim
            });

            const SSIMFunc = ssimModule.ssim;
            
            if (typeof SSIMFunc !== 'function') {
                throw new Error('SSIM function not available');
            }

            // Test setup
            const width = 8;
            const height = 8;
            const testBuffer = Buffer.alloc(width * height, 128);

            const testImage1 = {
                data: testBuffer,
                width: width,
                height: height,
                channels: 1
            };

            const testImage2 = {
                data: testBuffer,
                width: width,
                height: height,
                channels: 1
            };

            // Test SSIM function
            const testResult = SSIMFunc(testImage1, testImage2);
            
            if (testResult && typeof testResult === 'object' && typeof testResult.mssim === 'number') {
                logger.info('SSIM initialized successfully', {
                    service: "tatt2awai-bot",
                    mssim: testResult.mssim
                });
                ssim = SSIMFunc;
            } else {
                throw new Error('SSIM test failed - invalid result structure');
            }

        } catch (error) {
            logger.warn('SSIM initialization failed:', {
                error: error.message,
                stack: error.stack,
                service: "tatt2awai-bot"
            });
            ssim = null;
        }

        try {
            pixelmatch = (await import('pixelmatch')).default;
            logger.info('Pixelmatch initialized successfully');
        } catch (error) {
            logger.warn('Pixelmatch initialization failed:', error.message);
            pixelmatch = null;
        }
    },

    // Memory management helper
    async manageMemory(tempCache) {
        if (process.memoryUsage().heapUsed > 500 * 1024 * 1024) { // 500MB
            tempCache.clear();
            if (global.gc) global.gc();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    },

    compareQuickSignatures(sig1, sig2) {
        const aspectRatioMatch = Math.abs(sig1.aspectRatio - sig2.aspectRatio) < 
                            MATCH_CONSTANTS.ASPECT_RATIO_TOLERANCE;
        if (!aspectRatioMatch) return 0;

        const colorDiff = sig1.averageColor.reduce((diff, value, i) => 
            diff + Math.abs(value - sig2.averageColor[i]), 0) / sig1.averageColor.length;

        return 1 - (colorDiff / 255);
    },

// Helper functions for findExactMatch
    async generateImageSignature(buffer) {
        const metadata = await sharp(buffer).metadata();
        return {
            aspectRatio: metadata.width / metadata.height,
            size: metadata.size,
            format: metadata.format
        };
    },

    filterImageByMetadata(image, searchMetadata) {
        if (!image.path_lower.match(/\.(jpg|jpeg|png|webp)$/i)) return false;
        
        if (image.metadata?.width && image.metadata?.height) {
            const imageAspectRatio = image.metadata.width / image.metadata.height;
            const searchAspectRatio = searchMetadata.width / searchMetadata.height;
            return Math.abs(imageAspectRatio - searchAspectRatio) < MATCH_CONSTANTS.THRESHOLDS.ASPECT_RATIO;
        }

        return true;
    },

    compareQuickSignatures(sig1, sig2) {
        const aspectRatioMatch = Math.abs(sig1.aspectRatio - sig2.aspectRatio) < 0.1;
        if (!aspectRatioMatch) return 0;

        const colorDiff = sig1.averageColor.reduce((diff, value, i) => 
            diff + Math.abs(value - sig2.averageColor[i]), 0) / sig1.averageColor.length;

        return 1 - (colorDiff / 255);
    },

    async calculatePixelDifference(buffer1, buffer2) {
        const data1 = new Uint8Array(buffer1);
        const data2 = new Uint8Array(buffer2);
        let totalDiff = 0;
        const length = Math.min(data1.length, data2.length);

        for (let i = 0; i < length; i++) {
            totalDiff += Math.abs(data1[i] - data2[i]);
        }

        return totalDiff / length;
    }
};

// Export the complete package
const exportedMethods = {
    processImage: (...args) => imageProcessor.processImage(...args),
    getFileType: (...args) => imageProcessor.getFileType(...args),

 ImageCache: {
        get: (...args) => ImageCache.get(...args),
        cleanup: (...args) => ImageCache.cleanup(...args)
    },
    ImageHasher: {
        generateHash: (...args) => ImageHasher.generateHash(...args),
        generateMultiHash: (...args) => ImageHasher.generateMultiHash(...args),
        calculateSimilarity: (...args) => ImageHasher.calculateSimilarity(...args)
    },
    ImageSignatureGenerator: {
        generate: (...args) => ImageSignatureGenerator.generate(...args),
        generateDetailedColorHistogram: (...args) => ImageSignatureGenerator.generateDetailedColorHistogram(...args),
        generateAdvancedEdgeSignature: (...args) => ImageSignatureGenerator.generateAdvancedEdgeSignature(...args),
        generateBinarySignature: (...args) => ImageSignatureGenerator.generateBinarySignature(...args),
        generateAdvancedHash: (...args) => ImageSignatureGenerator.generateAdvancedHash(...args),
        createEdgeHistogram: (...args) => ImageSignatureGenerator.createEdgeHistogram(...args),
        calculateAdaptiveThreshold: (...args) => ImageSignatureGenerator.calculateAdaptiveThreshold(...args),
        generateWaveletHash: (...args) => ImageSignatureGenerator.generateWaveletHash(...args)
    },

    isImage: (...args) => imageProcessor.isImage(...args),
    detectTattooContent: (...args) => imageProcessor.detectTattooContent(...args),
    analyzeInkPatterns: (...args) => imageProcessor.analyzeInkPatterns(...args),
    analyzeSkinFeatures: (...args) => imageProcessor.analyzeSkinFeatures(...args),
    analyzeRemovalIndicators: (...args) => imageProcessor.analyzeRemovalIndicators(...args),
    detectEdges: (...args) => imageProcessor.detectEdges(...args),
    detectSkinTones: (...args) => imageProcessor.detectSkinTones(...args),
    isSkinTone: (...args) => imageProcessor.isSkinTone(...args),
    calculateSkinCoverage: (...args) => imageProcessor.calculateSkinCoverage(...args),
    calculateBoundingBoxArea: (...args) => imageProcessor.calculateBoundingBoxArea(...args),
    detectPatterns: (...args) => imageProcessor.detectPatterns(...args),
    analyzeColorProfile: (...args) => imageProcessor.analyzeColorProfile(...args),
    isInkLikeColor: (...args) => imageProcessor.isInkLikeColor(...args),
    detectFragmentationLevel: (...args) => imageProcessor.detectFragmentationLevel(...args),
    determineRemovalStage: (...args) => imageProcessor.determineRemovalStage(...args),
    validateImageFile: (...args) => imageProcessor.validateImageFile(...args),
    optimizeImage: (...args) => imageProcessor.optimizeImage(...args),
    initializeComparisonLibraries: (...args) => imageProcessor.initializeComparisonLibraries(...args),
    generateImageHash: (...args) => imageProcessor.generateImageHash(...args),
    generateFileHash: (...args) => imageProcessor.generateFileHash(...args),
    saveTempBuffer: (...args) => imageProcessor.saveTempBuffer(...args),
    analyzeTattooFeatures: (...args) => imageProcessor.analyzeTattooFeatures(...args),
    analyzeEdgeDefinition: (...args) => imageProcessor.analyzeEdgeDefinition(...args),
    analyzeSkinChanges: (...args) => imageProcessor.analyzeSkinChanges(...args),
    analyzeSkinReaction: (...args) => imageProcessor.analyzeSkinReaction(...args),
    analyzeFragmentation: (...args) => imageProcessor.analyzeFragmentation(...args),
    detectPatternChanges: (...args) => imageProcessor.detectPatternChanges(...args),
    detectBodyPartCombinations: (...args) => imageProcessor.detectBodyPartCombinations(...args),
    detectDotting: (...args) => imageProcessor.detectDotting(...args),
    detectGridLayout: (...args) => imageProcessor.detectGridLayout(...args),
    detectSequenceLayout: (...args) => imageProcessor.detectSequenceLayout(...args),
    detectChanges: (...args) => imageProcessor.detectChanges(...args),
    detectTimeBasedChanges: (...args) => imageProcessor.detectTimeBasedChanges(...args),
    enhancedCompareImages: (...args) => imageProcessor.enhancedCompareImages(...args),
    comparePixels: (...args) => imageProcessor.comparePixels(...args),
    compareSSIM: (...args) => imageProcessor.compareSSIM(...args),
    compareHash: (...args) => imageProcessor.compareHash(...args),
    compareColorHistograms: (...args) => imageProcessor.compareColorHistograms(...args),
    compareEdgeProfiles: (...args) => imageProcessor.compareEdgeProfiles(...args),
    compareInkDensity: (...args) => imageProcessor.compareInkDensity(...args),
    compareBlockHash: (...args) => imageProcessor.compareBlockHash(...args),
    compareColorProfiles: (...args) => imageProcessor.compareColorProfiles(...args),
    findExactMatch: (...args) => imageProcessor.findExactMatch(...args),
    validateMatch: (...args) => imageProcessor.validateMatch(...args),
    generateHash: (...args) => imageProcessor.generateHash(...args),
    findImageSequence: (...args) => imageProcessor.findImageSequence(...args),
    calculateHashSimilarity: (...args) => imageProcessor.calculateHashSimilarity(...args),
withTimeout: (...args) => imageProcessor.withTimeout(...args),
    calculateConfidence: (...args) => imageProcessor.calculateConfidence(...args),
    prefilterCandidates: (...args) => imageProcessor.prefilterCandidates(...args),
    promiseWithTimeout: (...args) => imageProcessor.promiseWithTimeout(...args),
    downloadWithRetry: (...args) => imageProcessor.downloadWithRetry(...args),
    extractImageFeatures: (...args) => imageProcessor.extractImageFeatures(...args),
    generateColorHistogram: (...args) => imageProcessor.generateColorHistogram(...args),
    generateEdgeSignature: (...args) => imageProcessor.generateEdgeSignature(...args),
    compareHashes: (...args) => imageProcessor.compareHashes(...args),
    compareHistograms: (...args) => imageProcessor.compareHistograms(...args),
    processInParallel: (...args) => imageProcessor.processInParallel(...args),
    quickVerify: (...args) => imageProcessor.quickVerify(...args),
    processMatch: (...args) => imageProcessor.processMatch(...args),
    verifyAndRankMatches: (...args) => imageProcessor.verifyAndRankMatches(...args),
    generateStandardThumbnail: (...args) => imageProcessor.generateStandardThumbnail(...args),
    generateThumbnail: (...args) => imageProcessor.generateThumbnail(...args),
    downloadAndProcess: (...args) => imageProcessor.downloadAndProcess(...args),
    compareQuick: (...args) => imageProcessor.compareQuick(...args),
    compareDetailed: (...args) => imageProcessor.compareDetailed(...args),
    processBatchWithLimit: (...args) => imageProcessor.processBatchWithLimit(...args),
    downloadAndNormalize: (...args) => imageProcessor.downloadAndNormalize(...args),
    prefilterBatch: (...args) => imageProcessor.prefilterBatch(...args),
    manageMemory: (...args) => imageProcessor.manageMemory(...args),
    calculatePixelDifference: (...args) => imageProcessor.calculatePixelDifference(...args),
    compareEdgePatterns: (...args) => imageProcessor.compareEdgePatterns(...args),
    calculateMethodConsistency: (...args) => imageProcessor.calculateMethodConsistency(...args),
    calculateScoreConsistency: (...args) => imageProcessor.calculateScoreConsistency(...args),
    calculateSignalStrength: (...args) => imageProcessor.calculateSignalStrength(...args),
    robustImageProcessing: (...args) => imageProcessor.robustImageProcessing(...args),
    compareColorMetrics: (...args) => imageProcessor.compareColorMetrics(...args),
    comparePerceptualHashes: (...args) => imageProcessor.comparePerceptualHashes(...args),
    compareQuickSignatures: (...args) => imageProcessor.compareQuickSignatures(...args),
    extractSequenceNumber: (...args) => imageProcessor.extractSequenceNumber(...args),
 compareBuffers: (...args) => imageProcessor.compareBuffers(...args),
    processWithTimeout: (...args) => imageProcessor.processWithTimeout(...args),   
comparePixelContent: (...args) => imageProcessor.comparePixelContent(...args),
processChunk: (...args) => imageProcessor.processChunk(...args),
retryDownload: (...args) => imageProcessor.retryDownload(...args),
generateDetailedThumb: (...args) => imageProcessor.generateDetailedThumb(...args),
getImageDetailedThumb: (...args) => imageProcessor.getImageDetailedThumb(...args),
compareEdgeContent: (...args) => imageProcessor.compareEdgeContent(...args),
verifyMatch: (...args) => imageProcessor.verifyMatch(...args),
generateVerificationThumb: (...args) => imageProcessor.generateVerificationThumb(...args),
 ImageSignatureGenerator
};

// Export everything as a single module
module.exports = {
    ...exportedMethods,
    SUPPORTED_IMAGE_TYPES,
    ANALYSIS_THRESHOLDS,
    CACHE_SETTINGS,
    IMAGE_PROCESSING,
    SEQUENCE_PATTERNS,
    SEQUENCE_KEYWORDS,
    TATTOO_KEYWORDS,
    COSMETIC_TATTOO_PATTERNS,
    BODY_PART_INDICATORS,
    ENHANCED_PATTERNS,
    MATCH_CONSTANTS
};
