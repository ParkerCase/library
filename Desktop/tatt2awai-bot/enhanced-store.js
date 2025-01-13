const { LRUCache } = require('lru-cache');
const pLimit = require('p-limit');
const path = require('path');
const fs = require('fs');
const fsp = fs.promises;
const sharp = require('sharp');
const logger = require('./logger');
const EnhancedSignatureGenerator = require('./enhanced-signature-generator');
const dropboxManager = require('./dropboxManager');

const SYSTEM_SETTINGS = {
    MATCHING: {
        EXACT_THRESHOLD: 0.95,
        SIMILAR_THRESHOLD: 0.85,
        MINI_HASH_THRESHOLD: 0.8,
        CONFIDENCE_THRESHOLD: 0.9,
        ASPECT_RATIO_TOLERANCE: 0.1
    },
    BATCH: {
        CONCURRENT_DOWNLOADS: 10,
        SIZE: 100,
        AUTH_CACHE_DURATION: 300,
        DOWNLOAD_TIMEOUT: 30000,
        SAVE_INTERVAL: 100  // Save every 100 signatures
    },
    MEMORY: {
        CACHE_TTL: 24 * 60 * 60 * 1000,  // 24 hours
        CACHE_MAX_SIZE: 10000,
        GC_THRESHOLD: 500 * 1024 * 1024  // 500MB
    }
};

const SIGNATURE_FILE = path.join(__dirname, 'image-signatures.json');
const BACKUP_DIR = path.join(__dirname, 'signature_backups');

class EnhancedSignatureStore {
    constructor() {
        this.cache = new LRUCache({
            max: SYSTEM_SETTINGS.MEMORY.CACHE_MAX_SIZE,
            ttl: SYSTEM_SETTINGS.MEMORY.CACHE_TTL
        });
        this.initialized = false;
        this.signatures = new Map();
        this.lastSaveTime = Date.now();
        this.initProgress = {
            total: 0,
            processed: 0,
            valid: 0,
            invalid: 0
        };

        // Ensure backup directory exists
        if (!fs.existsSync(BACKUP_DIR)) {
            fs.mkdirSync(BACKUP_DIR, { recursive: true });
        }
    }

    async initialize(entries) {
        if (this.initialized) {
            logger.info('Signature store already initialized');
            return true;
        }

        try {
            logger.info('Initializing signature store...');
            
            // Clear any existing signatures to prevent duplicates
            this.signatures.clear();
            
            // Create backup before loading
            await this.createBackup();
            
            // Load and validate existing signatures
            try {
                const signatureData = await fsp.readFile(SIGNATURE_FILE, 'utf8');
                const existingSignatures = JSON.parse(signatureData);
                
                // Validate and load signatures into memory
                let validCount = 0;
                let invalidCount = 0;
                let missingRequiredProps = 0;
                let malformedData = 0;
                
                Object.entries(existingSignatures).forEach(([path, signature]) => {
                    try {
                        if (this.isValidSignature(signature)) {
                            this.signatures.set(path, signature);
                            validCount++;
                        } else {
                            invalidCount++;
                            if (!this.hasRequiredProperties(signature)) {
                                missingRequiredProps++;
                            } else {
                                malformedData++;
                            }
                            logger.warn(`Invalid signature found for ${path}`, {
                                service: "tatt2awai-bot",
                                reason: this.getInvalidReason(signature)
                            });
                        }
                    } catch (error) {
                        invalidCount++;
                        logger.error(`Error processing signature for ${path}:`, {
                            service: "tatt2awai-bot",
                            error: error.message
                        });
                    }
                });
                
                this.initProgress = {
                    total: Object.keys(existingSignatures).length,
                    processed: validCount + invalidCount,
                    valid: validCount,
                    invalid: invalidCount
                };
                
                logger.info('Signature loading results:', {
                    service: "tatt2awai-bot",
                    totalEntriesInFile: Object.keys(existingSignatures).length,
                    validSignatures: validCount,
                    invalidSignatures: invalidCount,
                    missingProperties: missingRequiredProps,
                    malformedData: malformedData,
                    loadedIntoMemory: this.signatures.size
                });
                
            } catch (fileError) {
                logger.error('Failed to load signatures from file:', {
                    service: "tatt2awai-bot",
                    error: fileError.message,
                    stack: fileError.stack
                });
            }

            // Process new entries if provided
            if (entries?.length) {
                const missingEntries = entries.filter(entry => 
                    !this.signatures.has(entry.path_lower) &&
                    ['.jpg', '.jpeg', '.png', '.webp'].some(ext => 
                        entry.path_lower.toLowerCase().endsWith(ext)
                    )
                );

                if (missingEntries.length > 0) {
                    logger.info(`Processing ${missingEntries.length} missing signatures...`);
                    await this.processMissingEntries(missingEntries);
                }
            }

            this.initialized = true;
            logger.info('Signature store initialization complete', {
                service: "tatt2awai-bot",
                signatures: this.signatures.size,
                cacheSize: this.cache.size
            });

            return true;

        } catch (error) {
            logger.error('Failed to initialize signature store:', {
                service: "tatt2awai-bot",
                error: error.message,
                stack: error.stack
            });
            throw error;
        }
    }

    async processMissingEntries(entries) {
        const limit = pLimit(SYSTEM_SETTINGS.BATCH.CONCURRENT_DOWNLOADS);
        let processed = 0;
        let failed = 0;

        // Process in batches
        for (let i = 0; i < entries.length; i += SYSTEM_SETTINGS.BATCH.SIZE) {
            const batch = entries.slice(i, i + SYSTEM_SETTINGS.BATCH.SIZE);
            
            await Promise.all(batch.map(entry => limit(async () => {
                try {
                    const fileData = await dropboxManager.downloadFile(entry.path_lower);
                    if (!fileData?.result?.fileBinary) return;

                    const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                        fileData.result.fileBinary :
                        Buffer.from(fileData.result.fileBinary);

                    const signature = await EnhancedSignatureGenerator.generateSignature(buffer);
                    this.signatures.set(entry.path_lower, signature);
                    processed++;

                    // Save periodically
                    if (processed % SYSTEM_SETTINGS.BATCH.SAVE_INTERVAL === 0) {
                        await this.saveSignaturesToFile();
                        logger.info(`Processed and saved ${processed}/${entries.length} signatures`);
                    }

                    // Memory management
                    if (process.memoryUsage().heapUsed > SYSTEM_SETTINGS.MEMORY.GC_THRESHOLD) {
                        if (global.gc) {
                            global.gc();
                        }
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }

                } catch (error) {
                    failed++;
                    logger.error(`Failed to process ${entry.path_lower}:`, {
                        service: "tatt2awai-bot",
                        error: error.message,
                        stack: error.stack
                    });
                }
            })));

            logger.info('Batch processing progress:', {
                service: "tatt2awai-bot",
                processed,
                failed,
                total: entries.length,
                percentage: ((processed + failed) / entries.length * 100).toFixed(2)
            });
        }

        // Final save
        if (processed > 0) {
            await this.saveSignaturesToFile();
        }

        return { processed, failed };
    }

    async createBackup() {
        try {
            if (fs.existsSync(SIGNATURE_FILE)) {
                const backupPath = path.join(BACKUP_DIR, `signatures_backup_${Date.now()}.json`);
                await fsp.copyFile(SIGNATURE_FILE, backupPath);
                logger.info(`Created backup at ${backupPath}`);

                // Clean old backups (keep last 5)
                const backups = await fsp.readdir(BACKUP_DIR);
                if (backups.length > 5) {
                    const sortedBackups = backups
                        .filter(f => f.startsWith('signatures_backup_'))
                        .sort((a, b) => b.localeCompare(a));
                    
                    for (let i = 5; i < sortedBackups.length; i++) {
                        await fsp.unlink(path.join(BACKUP_DIR, sortedBackups[i]));
                    }
                }
            }
        } catch (error) {
            logger.error('Failed to create backup:', {
                service: "tatt2awai-bot",
                error: error.message
            });
        }
    }

    async saveSignaturesToFile() {
        const now = Date.now();
        try {
            await this.createBackup();
            
            const signatureObject = Object.fromEntries(this.signatures);
            await fsp.writeFile(
                SIGNATURE_FILE,
                JSON.stringify(signatureObject, null, 2)
            );
            
            this.lastSaveTime = now;
            logger.info(`Saved ${this.signatures.size} signatures to file`);
            
            return true;
        } catch (error) {
            logger.error('Failed to save signatures:', {
                service: "tatt2awai-bot",
                error: error.message,
                stack: error.stack,
                timeSinceLastSave: now - this.lastSaveTime
            });
            throw error;
        }
    }

    isValidSignature(signature) {
        return this.hasRequiredProperties(signature) && this.hasValidData(signature);
    }

    hasRequiredProperties(signature) {
        if (!signature || typeof signature !== 'object') return false;
        
        const requiredProps = [
            'perceptualHashes',
            'edgeSignature',
            'metadata',
            'signatureId'
        ];
        
        return requiredProps.every(prop => prop in signature);
    }

    hasValidData(signature) {
        try {
            // Check perceptual hashes
            if (!signature.perceptualHashes.mini?.binary || 
                !signature.perceptualHashes.normal?.binary) {
                return false;
            }

            // Check edge signature
            if (!Array.isArray(signature.edgeSignature.multi_scale) || 
                !signature.edgeSignature.dominant_orientation) {
                return false;
            }

            // Check metadata
            if (!signature.metadata?.dimensions?.aspectRatio) {
                return false;
            }

            return true;
        } catch (error) {
            return false;
        }
    }

    getInvalidReason(signature) {
        if (!signature || typeof signature !== 'object') {
            return 'Invalid signature object';
        }

        if (!this.hasRequiredProperties(signature)) {
            const missing = ['perceptualHashes', 'edgeSignature', 'metadata', 'signatureId']
                .filter(prop => !(prop in signature));
            return `Missing properties: ${missing.join(', ')}`;
        }

        if (!signature.perceptualHashes.mini?.binary || 
            !signature.perceptualHashes.normal?.binary) {
            return 'Invalid perceptual hashes';
        }

        if (!Array.isArray(signature.edgeSignature.multi_scale) || 
            !signature.edgeSignature.dominant_orientation) {
            return 'Invalid edge signature';
        }

        if (!signature.metadata?.dimensions?.aspectRatio) {
            return 'Invalid metadata';
        }

        return 'Unknown validation failure';
    }

    async findExactMatch(searchBuffer, allImages) {
        if (!this.initialized) {
            throw new Error('Signature store not initialized');
        }

        try {
            // Generate signature for search image
            const searchSignature = await EnhancedSignatureGenerator.generateSignature(searchBuffer);

            // Quick filtering pass using mini hashes
            const candidates = allImages.result.entries.filter(image => {
                const signature = this.signatures.get(image.path_lower);
                if (!signature) return false;

                const miniHashScore = this.compareHashes(
                    searchSignature.perceptualHashes.mini,
                    signature.perceptualHashes.mini
                );

                return miniHashScore > SYSTEM_SETTINGS.MATCHING.MINI_HASH_THRESHOLD;
            });

            logger.info(`Found ${candidates.length} potential matches`);

            let bestMatch = null;
            let bestScore = 0;

            for (const candidate of candidates) {
                const signature = this.signatures.get(candidate.path_lower);
                const similarity = await this.compareSignatures(searchSignature, signature);

                if (similarity.score > SYSTEM_SETTINGS.MATCHING.EXACT_THRESHOLD &&
                    similarity.confidence > SYSTEM_SETTINGS.MATCHING.CONFIDENCE_THRESHOLD) {
                    return {
                        path: candidate.path_lower,
                        score: similarity.score,
                        confidence: similarity.confidence,
                        isExact: true,
                        components: similarity.components
                    };
                }

                if (similarity.score > bestScore) {
                    bestScore = similarity.score;
                    bestMatch = {
                        path: candidate.path_lower,
                        score: similarity.score,
                        confidence: similarity.confidence,
                        isExact: false,
                        components: similarity.components
                    };
                }
            }

            return bestMatch;

        } catch (error) {
            logger.error('Error finding exact match:', {
                service: "tatt2awai-bot",
                error: error.message,
                stack: error.stack
            });
            throw error;
        }
    }

    compareHashes(hash1, hash2) {
        let diff = 0;
        for (let i = 0; i < hash1.binary.length; i++) {
            if (hash1.binary[i] !== hash2.binary[i]) diff++;
        }
        return 1 - (diff / hash1.binary.length);
    }

    async compareSignatures(sig1, sig2) {
        const scores = {
            perceptual: this.compareHashes(sig1.perceptualHashes.normal, sig2.perceptualHashes.normal),
            edge: this.compareEdgeSignatures(sig1.edgeSignature, sig2.edgeSignature),
            metadata: this.compareMetadata(sig1.metadata, sig2.metadata)
        };

        const weights = {
            perceptual: 0.6,
            edge: 0.3,
            metadata: 0.1
        };

        const score = Object.entries(scores).reduce((total, [key, value]) => 
            total + value * weights[key], 0);

        const confidence = this.calculateConfidence(scores);

        return { 
            score, 
            confidence,
            components: scores
        };
    }

compareEdgeSignatures(sig1, sig2) {
        const orientationMatch = Math.abs(
            sig1.dominant_orientation.angle - sig2.dominant_orientation.angle
        ) <= 45;

        const histogramSimilarity = sig1.multi_scale.reduce((sum, scale, idx) => {
            const correspondingScale = sig2.multi_scale[idx];
            if (!correspondingScale) return sum;

            return sum + this.compareHistograms(
                scale.signature,
                correspondingScale.signature
            );
        }, 0) / sig1.multi_scale.length;

        return orientationMatch ? histogramSimilarity : histogramSimilarity * 0.5;
    }

    compareHistograms(hist1, hist2) {
        let similarity = 0;
        for (let i = 0; i < hist1.length; i++) {
            similarity += Math.min(hist1[i], hist2[i]);
        }
        return similarity;
    }

    compareMetadata(meta1, meta2) {
        if (!meta1 || !meta2) return 0;
        
        const aspectRatioDiff = Math.abs(
            meta1.dimensions.aspectRatio - meta2.dimensions.aspectRatio
        );
        
        return aspectRatioDiff < SYSTEM_SETTINGS.MATCHING.ASPECT_RATIO_TOLERANCE ? 
            1 - (aspectRatioDiff / SYSTEM_SETTINGS.MATCHING.ASPECT_RATIO_TOLERANCE) : 0;
    }

    calculateConfidence(scores) {
        const values = Object.values(scores);
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        
        // Higher confidence if scores are consistently high with low variance
        const consistency = 1 - Math.sqrt(variance);
        const baseConfidence = mean * consistency;

        // Additional confidence boost if all scores are above threshold
        const allScoresHigh = values.every(val => val > SYSTEM_SETTINGS.MATCHING.SIMILAR_THRESHOLD);
        
        return allScoresHigh ? Math.min(baseConfidence * 1.2, 1) : baseConfidence;
    }

    getStatus() {
        return {
            initialized: this.initialized,
            signatureCount: this.signatures.size,
            cacheSize: this.cache.size,
            lastSaveTime: this.lastSaveTime,
            initProgress: this.initProgress,
            memoryUsage: {
                heapUsed: process.memoryUsage().heapUsed / (1024 * 1024),
                heapTotal: process.memoryUsage().heapTotal / (1024 * 1024)
            }
        };
    }

    async cleanup() {
        try {
            // Save any pending changes
            if (this.signatures.size > 0 && Date.now() - this.lastSaveTime > 60000) {
                await this.saveSignaturesToFile();
            }

            // Clear caches
            this.cache.clear();
            
            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }

            logger.info('Cleanup completed successfully', {
                service: "tatt2awai-bot",
                signatures: this.signatures.size,
                cacheSize: this.cache.size
            });
        } catch (error) {
            logger.error('Error during cleanup:', {
                service: "tatt2awai-bot",
                error: error.message,
                stack: error.stack
            });
        }
    }
}

module.exports = new EnhancedSignatureStore();
