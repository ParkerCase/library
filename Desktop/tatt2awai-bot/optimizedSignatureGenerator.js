const sharp = require('sharp');
const _ = require('lodash');
const path = require('path');
const { EnhancedSignatureGenerator } = require('./enhanced-signature-generator');
const BufferUtils = require('./bufferUtils');
const logger = require('./logger');
const { LRUCache } = require('lru-cache');
const crypto = require('crypto');

class SignatureOptimizer {
    constructor(config = {}) {
        this.config = {
            maxImageSize: config.maxImageSize || 2048,
            compressionQuality: config.compressionQuality || 90,
            maxFeaturePoints: config.maxFeaturePoints || 1000,
            cacheSize: config.cacheSize || 100,
            TEXTURE: {
                GABOR_ORIENTATIONS: config.TEXTURE?.GABOR_ORIENTATIONS || [0, 30, 60, 90, 120, 150],
                GABOR_FREQUENCIES: config.TEXTURE?.GABOR_FREQUENCIES || [0.1, 0.2, 0.3, 0.4],
                LBP_RADIUS: config.TEXTURE?.LBP_RADIUS || 2,
                GLCM_DISTANCES: config.TEXTURE?.GLCM_DISTANCES || [1, 2, 4]
            },
            COLOR: {
                COLOR_QUANT_LEVELS: {
                    RGB: config.COLOR?.COLOR_QUANT_LEVELS?.RGB || 64,
                    LAB: config.COLOR?.COLOR_QUANT_LEVELS?.LAB || 32,
                    HSV: config.COLOR?.COLOR_QUANT_LEVELS?.HSV || 48
                }
            }
        };

        this.featureCache = new LRUCache({
            max: this.config.cacheSize,
            maxSize: 5000,
            sizeCalculation: () => 1,
            ttl: 1000 * 60 * 60 // 1 hour
        });

        this.initializeOptimizer();
    }

    initializeOptimizer() {
        sharp.cache(false);
        sharp.concurrency(1);
        sharp.simd(true);
    }

    async preprocessImage(buffer) {
        if (!Buffer.isBuffer(buffer)) {
            throw new Error('Invalid input: Buffer expected');
        }

        try {
            const image = sharp(buffer, { failOnError: false });
            const metadata = await image.metadata();

            // Convert to standardized format
            return await image
                .rotate() // Auto-rotate based on EXIF
                .resize(this.config.maxImageSize, this.config.maxImageSize, {
                    fit: 'inside',
                    withoutEnlargement: true,
                    kernel: 'lanczos3'
                })
                .jpeg({
                    quality: this.config.compressionQuality,
                    chromaSubsampling: '4:4:4',
                    force: true,
                    mozjpeg: true
                })
                .toBuffer();
        } catch (error) {
            throw new Error(`Image preprocessing failed: ${error.message}`);
        }
    }
}

class OptimizedSignatureGenerator extends EnhancedSignatureGenerator {
    constructor(config = {}) {
        super(config);
        
        const mergedConfig = {
            ...config,
            TEXTURE: {
                ...this.config.TEXTURE,
                ...(config.TEXTURE || {})
            },
            COLOR: {
                ...this.config.COLOR,
                ...(config.COLOR || {})
            }
        };

        this.optimizer = new SignatureOptimizer(mergedConfig);
        this.setupProductionOptimizations();
        
        this.generateSignature = this.generateSignature.bind(this);
        this.generateOptimizedSignature = this.generateOptimizedSignature.bind(this);
    }

    setupProductionOptimizations() {
        sharp.cache(false);
        sharp.concurrency(1);
        sharp.simd(true);

        this.metrics = {
            generationTime: [],
            memoryUsage: [],
            successRate: 0,
            totalProcessed: 0
        };
    }

// optimizedSignatureGenerator.js
async generateSignature(buffer) {
    const startTime = process.hrtime.bigint();
    let processedBuffer = null;

    try {
        if (!Buffer.isBuffer(buffer)) {
            throw new Error('Invalid input: Buffer expected');
        }

        // Initial preprocessing
        const { buffer: validBuffer, metadata } = await this.validateAndPreprocessBuffer(buffer);
        processedBuffer = validBuffer;

        logger.info('Buffer preprocessing completed', {
            originalSize: buffer.length,
            processedSize: processedBuffer.length,
            format: metadata.format,
            dimensions: `${metadata.width}x${metadata.height}`
        });

        // Extract features with proper error handling
        const [colorFeatures, edgeFeatures, textureFeatures, hashFeatures] = await Promise.all([
            this.extractColorFeaturesWithQuality(processedBuffer).catch(err => {
                logger.error('Color feature extraction failed:', err);
                return this.getDefaultColorFeatures();
            }),
            this.extractEdgeFeaturesWithQuality(processedBuffer).catch(err => {
                logger.error('Edge feature extraction failed:', err);
                return this.getDefaultEdgeFeatures();
            }),
            this.extractTextureFeaturesWithQuality(processedBuffer).catch(err => {
                logger.error('Texture feature extraction failed:', err);
                return this.getDefaultTextureFeatures();
            }),
            this.generatePerceptualHashesWithQuality(processedBuffer).catch(err => {
                logger.error('Hash generation failed:', err);
                return this.getDefaultHashFeatures();
            })
        ]);

        const spatialFeatures = await this.generateSpatialVerificationWithQuality(processedBuffer);

        const signature = {
            version: "2.0",
            timestamp: Date.now(),
            metadata: {
                width: metadata.width,
                height: metadata.height,
                format: metadata.format,
                size: processedBuffer.length,
                processingTime: Number(process.hrtime.bigint() - startTime) / 1_000_000
            },
            colors: colorFeatures,
            edges: edgeFeatures,
            textures: textureFeatures,
            hashes: hashFeatures,
            spatial: spatialFeatures
        };

        signature.quality = await this.calculateSignatureQuality(signature);

        logger.info('Signature generation completed successfully', {
            processingTime: signature.metadata.processingTime,
            overallQuality: signature.quality.overall,
            featureQualities: {
                color: signature.colors?.quality?.overall || 0,
                edge: signature.edges?.quality?.overall || 0,
                texture: signature.textures?.quality?.overall || 0,
                hash: signature.hashes?.quality?.overall || 0,
                spatial: signature.spatial?.quality?.overall || 0
            }
        });

        return signature;

    } catch (error) {
        logger.error('Signature generation failed:', {
            error: error.message,
            stack: error.stack,
            bufferInfo: {
                isBuffer: Buffer.isBuffer(buffer),
                size: buffer?.length || 0
            }
        });
        return this.getDefaultSignature(error.message);
    } finally {
        if (processedBuffer && processedBuffer !== buffer) {
            processedBuffer = null;
        }
        if (global.gc) {
            global.gc();
        }
    }
}


logFeatureQualities(qualities) {
    logger.info('Feature Quality Analysis:', {
        color: {
            overall: qualities.color.overall,
            components: {
                dominantColors: qualities.color.components.dominant,
                distribution: qualities.color.components.distribution,
                harmony: qualities.color.components.harmony
            }
        },
        edge: {
            overall: qualities.edge.overall,
            components: {
                strength: qualities.edge.components.strength,
                continuity: qualities.edge.components.continuity,
                distribution: qualities.edge.components.distribution,
                junctions: qualities.edge.components.junctions
            }
        },
        texture: {
            overall: qualities.texture.overall,
            components: {
                glcm: qualities.texture.components.glcm,
                lbp: qualities.texture.components.lbp,
                gabor: qualities.texture.components.gabor
            }
        },
        hash: {
            overall: qualities.hash.overall,
            components: {
                robustness: qualities.hash.components.robustness,
                distinctiveness: qualities.hash.components.distinctiveness,
                stability: qualities.hash.components.stability
            }
        }
    });
}




async processWithTimeout(promise, timeoutMs = 30000, operation = 'Operation') {
    return Promise.race([
        promise,
        new Promise((_, reject) => {
            setTimeout(() => {
                reject(new Error(`${operation} timed out after ${timeoutMs}ms`));
            }, timeoutMs);
        })
    ]);
}

async processWithSharp(buffer, operation) {
    if (!buffer) {
        throw new Error('Invalid input: No buffer provided');
    }

    return this.processWithTimeout(
        (async () => {
            try {
                // First convert to a safe format
                const safeBuffer = await sharp(buffer, {
                    failOnError: false,
                    unlimited: true,
                    sequentialRead: true
                })
                .jpeg({
                    quality: 90,
                    chromaSubsampling: '4:4:4',
                    force: true
                })
                .toBuffer();

                // Then perform the requested operation
                const image = sharp(safeBuffer, {
                    failOnError: false,
                    unlimited: true,
                    sequentialRead: true
                });

                return await operation(image);
            } catch (error) {
                if (error.message.includes('unsupported image format')) {
                    // Try with more permissive options
                    const image = sharp(buffer, {
                        failOnError: false,
                        unlimited: true,
                        sequentialRead: true,
                        pages: 1
                    });
                    
                    // Force conversion to JPEG before operation
                    const safeImage = await image
                        .jpeg({ force: true })
                        .toBuffer();

                    return await operation(sharp(safeImage));
                }
                throw error;
            }
        })(),
        10000, // 10 second timeout for sharp operations
        'Sharp operation'
    );
}

    // Override the base validateAndPreprocessBuffer method
async validateAndPreprocessBuffer(buffer) {
    try {
        if (!Buffer.isBuffer(buffer)) {
            throw new Error('Invalid input: Buffer expected');
        }

        logger.info('Starting buffer preprocessing', {
            inputSize: buffer.length,
            isBuffer: Buffer.isBuffer(buffer)
        });

        const MAX_DIMENSION = 2048; // Set maximum dimension

        // First get metadata to check dimensions
        const metadata = await sharp(buffer, {
            failOnError: false,
            unlimited: true
        }).metadata();

        logger.info('Original image dimensions:', {
            width: metadata.width,
            height: metadata.height,
            format: metadata.format
        });

        // Calculate resize dimensions while maintaining aspect ratio
        let width = metadata.width;
        let height = metadata.height;
        if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
            const aspectRatio = width / height;
            if (width > height) {
                width = MAX_DIMENSION;
                height = Math.round(MAX_DIMENSION / aspectRatio);
            } else {
                height = MAX_DIMENSION;
                width = Math.round(MAX_DIMENSION * aspectRatio);
            }
        }

        // Convert to grayscale and standardize format
        const processed = await sharp(buffer, {
            failOnError: false,
            unlimited: true,
            sequentialRead: true,
            density: 300
        })
        .resize(width, height, {
            fit: 'inside',
            withoutEnlargement: true,
            kernel: 'lanczos3'
        })
        .grayscale()
        .jpeg({ quality: 90 })
        .toBuffer();

        const processedMetadata = await sharp(processed).metadata();

        if (!processedMetadata || !processedMetadata.width || !processedMetadata.height) {
            throw new Error('Invalid image metadata after preprocessing');
        }

        logger.info('Buffer preprocessing completed successfully', {
            originalSize: buffer.length,
            processedSize: processed.length,
            format: processedMetadata.format,
            originalDimensions: `${metadata.width}x${metadata.height}`,
            processedDimensions: `${processedMetadata.width}x${processedMetadata.height}`
        });

        return {
            buffer: processed,
            metadata: processedMetadata
        };

    } catch (error) {
        logger.error('Buffer preprocessing failed:', {
            error: error.message,
            stack: error.stack,
            bufferInfo: {
                isBuffer: Buffer.isBuffer(buffer),
                size: buffer?.length,
                type: typeof buffer
            }
        });
        throw error;
    }
}

async emergencyBufferRecovery(buffer) {
    try {
        // Try raw processing first
        const rawBuffer = await sharp(buffer, {
            failOnError: false,
            raw: {
                width: 1,
                height: 1,
                channels: 4
            }
        })
        .raw()
        .toBuffer();

        const processedBuffer = await sharp(rawBuffer, {
            raw: {
                width: 1,
                height: 1,
                channels: 4
            },
            failOnError: false
        })
        .jpeg({
            quality: 90,
            chromaSubsampling: '4:4:4',
            force: true
        })
        .toBuffer();

        // Get metadata from processed buffer
        const metadata = await sharp(processedBuffer, {
            failOnError: false
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error('Invalid metadata after recovery');
        }

        return {
            buffer: processedBuffer,
            metadata: {
                width: metadata.width,
                height: metadata.height,
                format: 'jpeg',
                channels: metadata.channels || 3,
                space: metadata.space || 'srgb'
            }
        };

    } catch (error) {
        // Last resort: try minimal conversion
        const minimalBuffer = await sharp(buffer, {
            failOnError: false,
            unlimited: true,
            sequentialRead: true
        })
        .jpeg({
            quality: 90,
            force: true
        })
        .toBuffer();

        const metadata = await sharp(minimalBuffer, {
            failOnError: false
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error('Emergency recovery failed');
        }

        return {
            buffer: minimalBuffer,
            metadata: {
                width: metadata.width,
                height: metadata.height,
                format: 'jpeg',
                channels: metadata.channels || 3,
                space: metadata.space || 'srgb'
            }
        };
    }
}

// Add helper method for safe buffer conversion


ensureJpegBuffer(buffer) {
    return new Promise(async (resolve, reject) => {
        try {
            if (!Buffer.isBuffer(buffer)) {
                throw new Error('Invalid input: Buffer expected');
            }

            const processedBuffer = await sharp(buffer, {
                failOnError: false,
                unlimited: true,
                sequentialRead: true
            })
            .jpeg({
                quality: 90,
                chromaSubsampling: '4:4:4',
                force: true
            })
            .toBuffer()
            .catch(async () => {
                // Fallback to PNG if JPEG fails
                return await sharp(buffer, {
                    failOnError: false,
                    unlimited: true,
                    sequentialRead: true
                })
                .png()
                .toBuffer();
            });

            // Verify the processed buffer
            const metadata = await sharp(processedBuffer, {
                failOnError: false
            }).metadata();

            if (!metadata || !metadata.width || !metadata.height) {
                throw new Error('Invalid processed buffer metadata');
            }

            resolve(processedBuffer);
        } catch (error) {
            reject(new Error(`Buffer conversion failed: ${error.message}`));
        }
    });
}

async generateSignature(buffer) {
    const startTime = process.hrtime.bigint();
    let processedBuffer = null;
    let sharpInstance = null;

    try {
        if (!Buffer.isBuffer(buffer)) {
            throw new Error('Invalid input: Buffer expected');
        }

        const cacheKey = this.calculateCacheKey(buffer);
        const cachedSignature = await this.checkCache(cacheKey);
        if (cachedSignature) {
            return cachedSignature;
        }

        // Process buffer with proper cleanup
        const { buffer: validBuffer, metadata } = await this.validateAndPreprocessBuffer(buffer);
        processedBuffer = validBuffer;

        // Extract features with proper timeouts and error handling
        const [colorFeatures, edgeFeatures, textureFeatures, hashFeatures] = await Promise.all([
            this.withTimeout(this.extractColorFeatures(processedBuffer), 10000),
            this.withTimeout(this.extractEdgeFeatures(processedBuffer), 10000),
            this.withTimeout(this.extractTextureFeatures(processedBuffer), 10000),
            this.withTimeout(this.generatePerceptualHashes(processedBuffer), 10000)
        ]);

        const spatialFeatures = await this.withTimeout(
            this.generateSpatialVerification(processedBuffer),
            15000
        );

        const signature = {
            version: "2.0",
            timestamp: Date.now(),
            metadata: {
                width: metadata.width,
                height: metadata.height,
                format: metadata.format,
                size: processedBuffer.length,
                processingTime: Number(process.hrtime.bigint() - startTime) / 1_000_000
            },
            colors: colorFeatures || this.getDefaultColorFeatures(),
            edges: edgeFeatures || this.getDefaultEdgeFeatures(),
            textures: textureFeatures || this.getDefaultTextureFeatures(),
            hashes: hashFeatures || this.getDefaultHashFeatures(),
            spatial: spatialFeatures || this.getDefaultSpatialFeatures()
        };

        signature.quality = await this.calculateSignatureQuality(signature);

        await this.cacheSignature(cacheKey, signature);

        return signature;

    } catch (error) {
        logger.error('Signature generation failed:', error);
        return this.getDefaultSignature(error.message);
    } finally {
        if (processedBuffer && processedBuffer !== buffer) {
            processedBuffer = null;
        }
        if (sharpInstance) {
            sharpInstance.destroy();
        }
        if (global.gc) {
            global.gc();
        }
    }
}

    generateOptimizedSignature(buffer) {
        return this.generateSignature(buffer);
    }

    getDefaultSignature(errorMessage = null) {
        return {
            version: "2.0",
            timestamp: Date.now(),
            metadata: {},
            colors: this.getDefaultColorFeatures(),
            edges: this.getDefaultEdgeFeatures(),
            textures: this.getDefaultTextureFeatures(),
            hashes: this.getDefaultHashFeatures(),
            spatial: this.getDefaultSpatialFeatures(),
            quality: this.getDefaultQuality(),
            error: errorMessage
        };
    }

    calculateCacheKey(buffer) {
        return crypto.createHash('sha256').update(buffer).digest('hex');
    }

    async checkCache(key) {
        return this.optimizer.featureCache.get(key);
    }

    async cacheSignature(key, signature) {
        this.optimizer.featureCache.set(key, signature);
        return signature;
    }

    recordMetrics(duration) {
        const durationMs = Number(duration) / 1_000_000;
        this.metrics.generationTime.push(durationMs);
        this.metrics.memoryUsage.push(process.memoryUsage().heapUsed);
        this.metrics.totalProcessed++;
        this.metrics.successRate = (this.metrics.successRate * (this.metrics.totalProcessed - 1) + 1) / this.metrics.totalProcessed;

        if (this.metrics.totalProcessed % 100 === 0) {
            logger.info('Signature generation metrics:', {
                averageTime: _.mean(this.metrics.generationTime),
                averageMemory: _.mean(this.metrics.memoryUsage) / (1024 * 1024) + 'MB',
                successRate: this.metrics.successRate,
                totalProcessed: this.metrics.totalProcessed
            });
        }
    }
}

module.exports = { OptimizedSignatureGenerator };
