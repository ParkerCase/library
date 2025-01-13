// signatureMigration.js
const path = require('path');
const fs = require('fs').promises;
const sharp = require('sharp');
const logger = require('./logger');
const dropboxManager = require('./dropboxManager');
const { OptimizedSignatureGenerator } = require('./optimizedSignatureGenerator');
const BufferUtils = require('./bufferUtils');

const pLimit = require('p-limit');
const os = require('os');

const TIMEOUTS = {
    DROPBOX_OPERATION: 120000,    // 2 minutes for Dropbox operations
    IMAGE_PROCESSING: 60000,      // 1 minute for image processing
    DEFAULT: 30000                // 30 seconds default
};


class TimeoutError extends Error {
    constructor(message) {
        super(message);
        this.name = 'TimeoutError';
    }
}

class SignatureMigration {
constructor(options = {}) {
        this.options = {
            batchSize: options.batchSize || 5,
            memoryConstrained: options.memoryConstrained || false,
            maxRetries: options.maxRetries || 3,
            timeouts: {
                dropbox: options.timeouts?.dropbox || TIMEOUTS.DROPBOX_OPERATION,
                processing: options.timeouts?.processing || TIMEOUTS.IMAGE_PROCESSING,
                default: options.timeouts?.default || TIMEOUTS.DEFAULT
            },
            ...options
        };

        this.progress = {
            total: 0,
            processed: 0,
            successful: 0,
            failed: 0,
            startTime: null,
            lastSave: null,
            errors: [],
            lastProcessedPath: null
        };

this.signatureGenerator = new OptimizedSignatureGenerator({
            maxImageSize: options.memoryConstrained ? 1024 : 2048,
            timeout: this.options.timeout
        });
    }


calculateTimeout(fileSize) {
    // Base timeout
    let timeout = this.options.timeouts.processing;
    
    // Adjust for file size
    if (fileSize) {
        const sizeMB = fileSize / (1024 * 1024);
        if (sizeMB > 5) { // For files larger than 5MB
            timeout = Math.min(
                timeout * 2,
                timeout * (1 + (sizeMB - 5) / 10)
            );
        }
    }
    
    return Math.min(timeout, 300000); // Cap at 5 minutes
}

validateSignatureComplexity(signature) {
    if (!signature || typeof signature !== 'object') return false;
    
    // Ensure all required components are present and non-empty
    const requiredComponents = ['colors', 'edges', 'textures', 'hashes'];
    const hasAllComponents = requiredComponents.every(component => {
        return signature[component] && Object.keys(signature[component]).length > 0;
    });
    
    if (!hasAllComponents) return false;
    
    // Check quality scores
    const qualityThreshold = 0.4;
    if (!signature.quality || signature.quality.overall < qualityThreshold) {
        return false;
    }
    
    return true;
}

async detectImageFormat(buffer) {
    try {
        const sharp = require('sharp');
        const metadata = await sharp(buffer).metadata();
        return metadata.format;
    } catch (error) {
        logger.error('Error detecting image format:', error);
        return null;
    }
}

isSupportedFormat(format) {
    const supportedFormats = new Set([
        'jpeg', 'jpg', 'png', 'webp', 
        'tiff', 'gif', 'svg', 'heic'
    ]);
    return format && supportedFormats.has(format.toLowerCase());
}

async validateBuffer(buffer) {
    if (!Buffer.isBuffer(buffer)) {
        logger.error('Invalid input type provided:', {
            type: typeof buffer,
            isBuffer: Buffer.isBuffer(buffer)
        });
        throw new Error('Invalid input: Buffer expected');
    }

    if (buffer.length === 0) {
        logger.error('Empty buffer provided');
        throw new Error('Empty buffer');
    }

    try {
        // Verify buffer can be processed by Sharp
        const metadata = await sharp(buffer, {
            failOnError: false,
            limitInputPixels: this.options.maxImageSize * this.options.maxImageSize,
            sequentialRead: true
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error('Invalid image metadata');
        }

        logger.info('Buffer validation successful:', {
            format: metadata.format,
            width: metadata.width,
            height: metadata.height,
            size: Math.round(buffer.length / 1024) + 'KB'
        });

        return {
            buffer,
            metadata,
            isValid: true
        };
    } catch (error) {
        logger.error('Buffer validation failed:', {
            error: error.message,
            stack: error.stack,
            bufferSize: buffer.length
        });
        
        throw new Error(`Buffer validation failed: ${error.message}`);
    }
}

async validate() {
        try {
            logger.info('Starting validation...');
            // Use longer timeout for Dropbox operations
            const entries = await this.processWithTimeout(
                dropboxManager.fetchDropboxEntries(''),
                this.options.timeouts.dropbox
            );
            
            const imageFiles = entries.result.entries.filter(entry =>
                ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'].some(ext =>
                    entry.path_lower.endsWith(ext)
                )
            );

            if (imageFiles.length === 0) {
                throw new Error('No valid images found for migration');
            }

            logger.info(`Found ${imageFiles.length} images for processing`);

            // Validate memory availability before processing
            const memInfo = process.memoryUsage();
            logger.info('Memory status before validation:', {
                heapUsed: Math.round(memInfo.heapUsed / 1024 / 1024) + 'MB',
                heapTotal: Math.round(memInfo.heapTotal / 1024 / 1024) + 'MB'
            });

            const sampleSize = Math.min(5, imageFiles.length);
            const sample = imageFiles
                .sort(() => 0.5 - Math.random())
                .slice(0, sampleSize);

            logger.info(`Testing ${sampleSize} sample images...`);

            let successful = 0;
            for (const entry of sample) {
                try {
                    // Force cleanup before each sample
                    if (global.gc) {
                        global.gc();
                    }
                    
                    // Add delay between samples to prevent memory buildup
                    if (successful > 0) {
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }

                    await this.processImage(entry, 1);
                    successful++;
                    logger.info(`Sample ${successful}/${sampleSize} processed successfully`);
                    
                    // Log memory usage after each successful sample
                    const currentMem = process.memoryUsage();
                    logger.info('Memory usage after sample:', {
                        heapUsed: Math.round(currentMem.heapUsed / 1024 / 1024) + 'MB'
                    });
                } catch (error) {
                    logger.error('Validation sample failed:', {
                        path: entry.path_lower,
                        error: error.message,
                        stack: error.stack
                    });
                    
                    // Enhanced error logging for unsupported format errors
                    if (error.message.includes('unsupported image format')) {
                        logger.error('Unsupported format details:', {
                            path: entry.path_lower,
                            format: entry.path_lower.split('.').pop()
                        });
                    }
                }
            }

            // More stringent validation criteria
            const validationSuccess = successful / sampleSize >= 0.8;
            logger.info('Validation completed', {
                success: validationSuccess,
                sampleSize,
                successfulSamples: successful,
                failureRate: ((sampleSize - successful) / sampleSize * 100).toFixed(2) + '%'
            });

            return validationSuccess;
        } catch (error) {
            logger.error('Validation failed:', {
                error: error.message,
                stack: error.stack,
                phase: 'validation'
            });
            error.phase = 'validation';
            throw error;
        } finally {
            // Ensure cleanup after validation
            await this.optimizeMemory();
        }
    }

async preprocessBuffer(buffer, originalFormat) {
    const startTime = Date.now();
    let processedBuffer = buffer;
    const sharp = require('sharp');
    
    try {
        logger.info('Starting buffer preprocessing...', {
            originalFormat,
            inputSize: Math.round(buffer.length / 1024) + 'KB'
        });

        // Initial normalization to ensure consistent format
        processedBuffer = await sharp(buffer, { failOnError: false })
            .rotate() // Auto-rotate based on EXIF
            .withMetadata() // Preserve metadata
            .toBuffer();

        // Get metadata after initial normalization
        const metadata = await sharp(processedBuffer, { failOnError: false }).metadata();
        
        // Resize if needed
        if (metadata.width > 2048 || metadata.height > 2048) {
            processedBuffer = await sharp(processedBuffer, { failOnError: false })
                .resize(2048, 2048, {
                    fit: 'inside',
                    withoutEnlargement: true
                })
                .toBuffer();
        }

        // Convert to standardized JPEG format
        processedBuffer = await sharp(processedBuffer, { failOnError: false })
            .jpeg({
                quality: 90,
                chromaSubsampling: '4:4:4',
                force: true // Force JPEG output
            })
            .toBuffer();

        // Verify processed buffer
        const processedMetadata = await sharp(processedBuffer, { failOnError: false }).metadata();
        
        if (!processedMetadata || !processedMetadata.width || !processedMetadata.height) {
            throw new Error('Invalid processed buffer metadata');
        }

        // Log success
        logger.info('Buffer preprocessing successful:', {
            originalFormat,
            finalFormat: processedMetadata.format,
            originalSize: Math.round(buffer.length / 1024) + 'KB',
            processedSize: Math.round(processedBuffer.length / 1024) + 'KB',
            dimensions: `${processedMetadata.width}x${processedMetadata.height}`,
            duration: Date.now() - startTime + 'ms'
        });

        return processedBuffer;

    } catch (error) {
        logger.error('Buffer preprocessing failed:', {
            error: error.message,
            stack: error.stack,
            format: originalFormat,
            duration: Date.now() - startTime + 'ms'
        });

        // Try emergency fallback conversion
        try {
            processedBuffer = await sharp(buffer, { 
                failOnError: false,
                unlimited: true,
                sequentialRead: true
            })
            .jpeg({ 
                quality: 85,
                chromaSubsampling: '4:4:4',
                force: true
            })
            .toBuffer();

            const fallbackMetadata = await sharp(processedBuffer, { failOnError: false }).metadata();
            if (fallbackMetadata && fallbackMetadata.width && fallbackMetadata.height) {
                logger.info('Fallback conversion successful');
                return processedBuffer;
            }
        } catch (fallbackError) {
            logger.error('Fallback conversion also failed:', fallbackError);
        }

        throw error;
    } finally {
        // Ensure Sharp instance is cleaned up
        if (processedBuffer !== buffer) {
            if (!this._tempBuffers) this._tempBuffers = [];
            this._tempBuffers.push(processedBuffer);
        }
    }
}

// Helper function to detect image corruption
async isImageCorrupted(buffer) {
    try {
        const sharp = require('sharp');
        const metadata = await sharp(buffer, { failOnError: true }).metadata();
        return !(metadata && metadata.width && metadata.height);
    } catch (error) {
        return true;
    }
}

// Helper function to optimize image quality settings based on size
 getOptimalQualitySettings(fileSize) {
    const sizeMB = fileSize / (1024 * 1024);
    if (sizeMB > 10) {
        return { quality: 85, chromaSubsampling: '4:2:0' };
    } else if (sizeMB > 5) {
        return { quality: 90, chromaSubsampling: '4:2:2' };
    }
    return { quality: 90, chromaSubsampling: '4:4:4' };
}


async ensureJpegFormat(buffer) {
    const sharp = require('sharp');
    try {
        // First try to get metadata
        const metadata = await sharp(buffer, { failOnError: false }).metadata();
        
        // Prepare consistent sharp options
        const sharpOptions = {
            failOnError: false,
            unlimited: true,
            sequentialRead: true,
            density: 300
        };

        // Prepare consistent JPEG options
        const jpegOptions = {
            quality: 90,
            chromaSubsampling: '4:4:4',
            force: true,
            mozjpeg: true,
            optimizeCoding: true,
            trellisQuantisation: true,
            overshootDeringing: true,
            optimizeScans: true
        };

        // If already JPEG, just ensure proper options are applied
        if (metadata && metadata.format === 'jpeg') {
            return await sharp(buffer, sharpOptions)
                .jpeg(jpegOptions)
                .toBuffer();
        }

        // For non-JPEG, do full conversion with preprocessing
        const converted = await sharp(buffer, sharpOptions)
            .rotate() // Auto-rotate based on EXIF
            .resize(2048, 2048, { 
                fit: 'inside',
                withoutEnlargement: true,
                kernel: 'lanczos3'
            })
            .removeAlpha()
            .jpeg(jpegOptions)
            .toBuffer();

        // Verify conversion was successful
        const verifyMetadata = await sharp(converted, { failOnError: false }).metadata();
        if (!verifyMetadata || verifyMetadata.format !== 'jpeg') {
            throw new Error('JPEG conversion verification failed');
        }

        // Verify the buffer is valid
        if (!Buffer.isBuffer(converted) || converted.length === 0) {
            throw new Error('Invalid converted buffer');
        }

        return converted;
    } catch (error) {
        logger.error('JPEG conversion failed:', {
            error: error.message,
            stack: error.stack,
            inputFormat: metadata?.format || 'unknown',
            inputSize: buffer?.length || 0
        });
        throw new Error(`JPEG conversion failed: ${error.message}`);
    }
}

async processImage(entry, attempt = 1) {
    const startTime = Date.now();
    let buffer = null;
    let processedBuffer = null;
    
    try {
        logger.info(`Processing image: ${entry.path_display} (attempt ${attempt})`);
        
        // Download file
        const fileData = await this.processWithTimeout(
            dropboxManager.downloadFile(entry.path_lower),
            this.options.timeouts.dropbox,
            'File download'
        );

        if (!fileData?.result?.fileBinary) {
            throw new Error('No file data received from Dropbox');
        }

        // Convert fileBinary to Buffer properly
        buffer = Buffer.isBuffer(fileData.result.fileBinary) 
            ? fileData.result.fileBinary 
            : Buffer.from(fileData.result.fileBinary);

        logger.info('Initial buffer created', {
            size: buffer.length,
            isBuffer: Buffer.isBuffer(buffer)
        });

        // Initial preprocessing with robust error handling
        const preprocessed = await sharp(buffer, {
            failOnError: false,
            unlimited: true,
            sequentialRead: true,
            density: 300,
            pages: 1
        })
        .rotate() // Auto-rotate based on EXIF
        .resize(2048, 2048, {
            fit: 'inside',
            withoutEnlargement: true,
            kernel: 'lanczos3'
        })
        .jpeg({
            quality: 90,
            chromaSubsampling: '4:4:4',
            force: true
        })
        .toBuffer()
        .catch(async (err) => {
            logger.warn('JPEG conversion failed, trying PNG:', err);
            return sharp(buffer, {
                failOnError: false,
                unlimited: true,
                sequentialRead: true
            })
            .png()
            .toBuffer()
            .catch(async (err2) => {
                logger.warn('PNG conversion failed, trying WebP:', err2);
                return sharp(buffer, {
                    failOnError: false,
                    unlimited: true,
                    sequentialRead: true
                })
                .webp()
                .toBuffer();
            });
        });

        if (!preprocessed) {
            throw new Error('Image preprocessing failed');
        }

        // Verify the preprocessed buffer
        const metadata = await sharp(preprocessed, {
            failOnError: false
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error('Invalid image metadata after preprocessing');
        }

        logger.info('Image preprocessing successful', {
            originalSize: buffer.length,
            processedSize: preprocessed.length,
            format: metadata.format,
            dimensions: `${metadata.width}x${metadata.height}`
        });

        // Generate signature with preprocessed buffer
        const signature = await this.processWithTimeout(
            this.signatureGenerator.generateSignature(preprocessed),
            this.calculateTimeout(entry.size),
            'Signature generation'
        );

        logger.info('Signature generation quality metrics:', {
            path: entry.path_display,
            quality: signature.quality,
            features: {
                colors: signature.colors?.quality || 'missing',
                edges: signature.edges?.quality || 'missing',
                textures: signature.textures?.quality || 'missing',
                hashes: signature.hashes?.quality || 'missing',
                spatial: signature.spatial?.quality || 'missing'
            },
            processingTime: Date.now() - startTime,
            success: true
        });

        await this.updateProgress(entry, 'success');
        return signature;

    } catch (error) {
        logger.error('Image processing failed:', {
            path: entry.path_display,
            attempt,
            error: error.message,
            stack: error.stack,
            bufferInfo: buffer ? {
                isBuffer: Buffer.isBuffer(buffer),
                size: buffer.length,
                type: typeof buffer
            } : 'No buffer created'
        });

        if (attempt < this.options.maxRetries) {
            const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
            await new Promise(resolve => setTimeout(resolve, delay));
            return this.processImage(entry, attempt + 1);
        }
        
        await this.updateProgress(entry, 'failed');
        throw error;
    } finally {
        // Clean up
        buffer = null;
        processedBuffer = null;
        
        if (global.gc) {
            global.gc();
        }
    }
}

async processWithBufferRecovery(buffer) {
    let attempts = 0;
    const maxAttempts = this.options.maxRetries || 3;

    while (attempts < maxAttempts) {
        try {
            attempts++;
            logger.info(`Processing attempt ${attempts}/${maxAttempts}`);

            // Try to validate and process the buffer
            const validationResult = await this.validateBuffer(buffer);
            
            if (!validationResult.isValid) {
                throw new Error('Buffer validation failed');
            }

            // Process the validated buffer
            const processed = await this.processBuffer(validationResult.buffer, validationResult.metadata);
            
            logger.info('Buffer processing successful:', {
                attempt: attempts,
                format: processed.format,
                width: processed.width,
                height: processed.height
            });

            return processed;

        } catch (error) {
            logger.error(`Processing attempt ${attempts} failed:`, {
                error: error.message,
                stack: error.stack
            });

            if (attempts === maxAttempts) {
                throw new Error(`All processing attempts failed: ${error.message}`);
            }

            // Try recovery before next attempt
            try {
                buffer = await this.attemptBufferRecovery(buffer);
                await this.optimizeMemory();
                await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
            } catch (recoveryError) {
                logger.error('Recovery attempt failed:', recoveryError);
            }
        }
    }

    throw new Error('Buffer processing failed after all attempts');
}

async attemptBufferRecovery(buffer) {
    logger.info('Attempting buffer recovery');
    
    try {
        // First try: Basic JPEG conversion
        const processed = await sharp(buffer, {
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

        // Verify the recovered buffer
        const metadata = await sharp(processed, { failOnError: false }).metadata();
        if (metadata?.width && metadata?.height) {
            logger.info('Buffer recovery successful using JPEG conversion');
            return processed;
        }

        // Second try: PNG conversion
        const pngProcessed = await sharp(buffer, {
            failOnError: false,
            unlimited: true
        })
        .png({
            compressionLevel: 9,
            force: true
        })
        .toBuffer();

        const pngMetadata = await sharp(pngProcessed, { failOnError: false }).metadata();
        if (pngMetadata?.width && pngMetadata?.height) {
            logger.info('Buffer recovery successful using PNG conversion');
            return pngProcessed;
        }

        // Third try: Raw processing with minimum options
        const rawProcessed = await sharp(buffer, {
            failOnError: false,
            unlimited: true,
            sequentialRead: true,
            raw: {
                width: 1,
                height: 1,
                channels: 4
            }
        })
        .raw()
        .toBuffer();

        if (rawProcessed.length > 0) {
            logger.info('Buffer recovery successful using raw processing');
            return rawProcessed;
        }

        throw new Error('All recovery attempts failed');

    } catch (error) {
        logger.error('Buffer recovery failed:', {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
}

async processWithTimeout(promise, timeoutMs, operation = 'Unknown operation') {
    let timeoutId;
    const startTime = Date.now();
    const memBefore = process.memoryUsage();
    
    try {
        logger.info(`Starting ${operation}...`);
        
        const result = await Promise.race([
            promise.then(value => {
                const duration = Date.now() - startTime;
                const memAfter = process.memoryUsage();
                
                logger.info(`${operation} completed in ${duration}ms`, {
                    memoryDelta: {
                        heap: Math.round((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024) + 'MB',
                        rss: Math.round((memAfter.rss - memBefore.rss) / 1024 / 1024) + 'MB'
                    }
                });

                // Check for significant memory increases
                const memoryIncrease = memAfter.heapUsed - memBefore.heapUsed;
                if (memoryIncrease > 100 * 1024 * 1024) { // 100MB threshold
                    logger.warn(`Large memory increase detected in ${operation}`, {
                        memoryIncrease: Math.round(memoryIncrease / 1024 / 1024) + 'MB'
                    });
                }

                return value;
            }),
            new Promise((_, reject) => {
                timeoutId = setTimeout(() => {
                    const error = new TimeoutError(`Operation '${operation}' timed out after ${timeoutMs}ms`);
                    error.timeoutMs = timeoutMs;
                    error.duration = Date.now() - startTime;
                    error.memoryUsage = process.memoryUsage();
                    error.operation = operation;
                    
                    logger.error('Operation timeout details:', {
                        operation,
                        timeoutMs,
                        duration: error.duration,
                        memoryUsage: {
                            heapUsed: Math.round(error.memoryUsage.heapUsed / 1024 / 1024) + 'MB',
                            heapTotal: Math.round(error.memoryUsage.heapTotal / 1024 / 1024) + 'MB',
                            rss: Math.round(error.memoryUsage.rss / 1024 / 1024) + 'MB'
                        },
                        memoryDelta: {
                            heap: Math.round((error.memoryUsage.heapUsed - memBefore.heapUsed) / 1024 / 1024) + 'MB',
                            rss: Math.round((error.memoryUsage.rss - memBefore.rss) / 1024 / 1024) + 'MB'
                        }
                    });

                    // Attempt emergency cleanup on timeout
                    try {
                        if (global.gc) {
                            global.gc();
                            logger.info('Performed emergency garbage collection after timeout');
                        }
                    } catch (cleanupError) {
                        logger.error('Failed to perform emergency cleanup:', cleanupError);
                    }

                    reject(error);
                }, timeoutMs);
            })
        ]);

        return result;

    } catch (error) {
        if (!(error instanceof TimeoutError)) {
            logger.error(`Error in ${operation}:`, {
                error: error.message,
                stack: error.stack,
                duration: Date.now() - startTime,
                memoryUsage: process.memoryUsage()
            });
        }
        throw error;
    } finally {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }

        // Monitor operations taking longer than expected
        const duration = Date.now() - startTime;
        if (duration > timeoutMs * 0.8) {
            logger.warn(`Operation '${operation}' took ${duration}ms (${Math.round(duration/timeoutMs*100)}% of timeout)`, {
                timeoutMs,
                duration,
                operation
            });
        }

        // Check final memory state
        const memAfter = process.memoryUsage();
        const memDelta = {
            heap: Math.round((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024),
            rss: Math.round((memAfter.rss - memBefore.rss) / 1024 / 1024)
        };

        if (memDelta.heap > 50) { // 50MB threshold
            logger.warn(`High memory usage detected in ${operation}`, {
                memoryDelta: memDelta
            });
            
            if (global.gc) {
                global.gc();
                logger.info('Performed garbage collection due to high memory usage');
                
                // Log memory state after garbage collection
                const memAfterGC = process.memoryUsage();
                logger.info('Memory state after garbage collection:', {
                    heapUsed: Math.round(memAfterGC.heapUsed / 1024 / 1024) + 'MB',
                    recovered: Math.round((memAfter.heapUsed - memAfterGC.heapUsed) / 1024 / 1024) + 'MB'
                });
            }
        }
    }
}

async cleanupResources(buffer = null) {
    const startTime = Date.now();
    const initialMemory = process.memoryUsage();
    
    try {
        logger.info('Starting resource cleanup...', {
            initialMemory: {
                heapUsed: Math.round(initialMemory.heapUsed / 1024 / 1024) + 'MB',
                rss: Math.round(initialMemory.rss / 1024 / 1024) + 'MB'
            }
        });

        // Track what we're cleaning
        const cleanupOperations = {
            buffer: false,
            sharp: false,
            signatureGenerator: false,
            tempBuffers: false,
            gc: false
        };

        // 1. Clean up provided buffer
        if (buffer && Buffer.isBuffer(buffer)) {
            try {
                buffer.fill(0);
                buffer = null;
                cleanupOperations.buffer = true;
            } catch (bufferError) {
                logger.warn('Buffer cleanup failed:', bufferError);
            }
        }

        // 2. Reset Sharp module state
        try {
            const sharp = require('sharp');
            sharp.cache(false);
            sharp.concurrency(1);
            await sharp.cache({ items: 0 });
            cleanupOperations.sharp = true;
        } catch (sharpError) {
            logger.warn('Sharp cleanup failed:', sharpError);
        }

        // 3. Clean up signature generator resources
        if (this.signatureGenerator) {
            try {
                await this.signatureGenerator.cleanupTemporaryResources();
                this.signatureGenerator = null;
                cleanupOperations.signatureGenerator = true;
            } catch (sigGenError) {
                logger.warn('Signature generator cleanup failed:', sigGenError);
            }
        }

        // 4. Clean up temporary buffers
        if (this._tempBuffers && this._tempBuffers.length) {
            try {
                this._tempBuffers.forEach(tempBuffer => {
                    if (tempBuffer && Buffer.isBuffer(tempBuffer)) {
                        tempBuffer.fill(0);
                    }
                });
                this._tempBuffers = [];
                cleanupOperations.tempBuffers = true;
            } catch (tempBufferError) {
                logger.warn('Temporary buffers cleanup failed:', tempBufferError);
            }
        }

        // 5. Force garbage collection if available
        if (global.gc) {
            try {
                // Wait a short time to allow any pending operations to complete
                await new Promise(resolve => setTimeout(resolve, 100));
                global.gc();
                cleanupOperations.gc = true;
            } catch (gcError) {
                logger.warn('Garbage collection failed:', gcError);
            }
        }

        // Check memory state after cleanup
        const finalMemory = process.memoryUsage();
        const memoryReduction = {
            heap: Math.round((initialMemory.heapUsed - finalMemory.heapUsed) / 1024 / 1024),
            rss: Math.round((initialMemory.rss - finalMemory.rss) / 1024 / 1024)
        };

        // Verify cleanup effectiveness
        const cleanupSuccess = Object.values(cleanupOperations).every(op => op);
        if (!cleanupSuccess) {
            logger.warn('Some cleanup operations failed:', {
                operations: cleanupOperations
            });
        }

        // Additional garbage collection if memory reduction is insufficient
        if (memoryReduction.heap < 10 && global.gc) { // Less than 10MB reduction
            logger.info('Insufficient memory reduction, performing additional GC');
            await new Promise(resolve => setTimeout(resolve, 200));
            global.gc();
            
            const finalCheck = process.memoryUsage();
            memoryReduction.heap = Math.round((initialMemory.heapUsed - finalCheck.heapUsed) / 1024 / 1024);
            memoryReduction.rss = Math.round((initialMemory.rss - finalCheck.rss) / 1024 / 1024);
        }

        logger.info('Resource cleanup completed:', {
            duration: Date.now() - startTime + 'ms',
            memoryReduction,
            operations: cleanupOperations
        });

        return {
            success: cleanupSuccess,
            memoryReduction,
            operations: cleanupOperations,
            duration: Date.now() - startTime
        };

    } catch (error) {
        logger.error('Resource cleanup failed:', {
            error: error.message,
            stack: error.stack
        });

        // Emergency cleanup attempt
        try {
            if (global.gc) {
                global.gc();
                logger.info('Performed emergency garbage collection after cleanup failure');
            }
        } catch (emergencyError) {
            logger.error('Emergency cleanup also failed:', emergencyError);
        }

        throw error;
    }
}

checkColorSpaceIssues(metadata) {
    const issues = [];
    const recommendedSpaces = new Set(['srgb', 'rgb', 'gray']);

    if (metadata.space && !recommendedSpaces.has(metadata.space.toLowerCase())) {
        issues.push({
            type: 'color_space',
            current: metadata.space,
            recommended: 'sRGB'
        });
    }

    if (metadata.channels) {
        if (metadata.channels > 4) {
            issues.push({
                type: 'channels',
                current: metadata.channels,
                recommended: 4
            });
        }
    }

    if (metadata.depth) {
        if (metadata.depth !== 8 && metadata.depth !== 16) {
            issues.push({
                type: 'bit_depth',
                current: metadata.depth,
                recommended: 8
            });
        }
    }

    return issues.length > 0 ? issues : null;
}

async verifyBufferState(buffer, stage) {
    const startTime = Date.now();
    const memoryBefore = process.memoryUsage();

    try {
        if (!Buffer.isBuffer(buffer)) {
            throw new Error(`Invalid buffer type at ${stage}`);
        }

        if (buffer.length === 0) {
            throw new Error(`Empty buffer at ${stage}`);
        }

        // Get metadata using Sharp with error handling
        const metadata = await sharp(buffer, {
            failOnError: false,
            unlimited: true,
            sequentialRead: true,
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error(`Invalid image metadata at ${stage}`);
        }

        // Verify basic image processing capability
        await sharp(buffer, { failOnError: false })
            .resize(10, 10) // Small resize to verify processability
            .toBuffer();

        // Check for color space issues
        const colorSpaceIssues = this.checkColorSpaceIssues(metadata);
        if (colorSpaceIssues) {
            logger.warn(`Color space issues detected at ${stage}:`, colorSpaceIssues);
        }

        // Calculate memory impact
        const memoryAfter = process.memoryUsage();
        const memoryImpact = {
            heapDelta: Math.round((memoryAfter.heapUsed - memoryBefore.heapUsed) / 1024 / 1024),
            rssDelta: Math.round((memoryAfter.rss - memoryBefore.rss) / 1024 / 1024)
        };

        logger.info(`Buffer verification at ${stage} successful:`, {
            format: metadata.format,
            dimensions: `${metadata.width}x${metadata.height}`,
            channels: metadata.channels,
            space: metadata.space,
            size: Math.round(buffer.length / 1024) + 'KB',
            duration: Date.now() - startTime + 'ms',
            memoryImpact
        });

        return {
            isValid: true,
            metadata,
            colorSpaceIssues,
            memoryImpact,
            verificationTime: Date.now() - startTime
        };

    } catch (error) {
        logger.error(`Buffer verification failed at ${stage}:`, {
            error: error.message,
            stack: error.stack,
            bufferSize: buffer?.length,
            duration: Date.now() - startTime
        });

        return {
            isValid: false,
            error: error.message,
            stage,
            memoryImpact: {
                heapDelta: Math.round((process.memoryUsage().heapUsed - memoryBefore.heapUsed) / 1024 / 1024),
                rssDelta: Math.round((process.memoryUsage().rss - memoryBefore.rss) / 1024 / 1024)
            },
            verificationTime: Date.now() - startTime
        };
    }
}

// Helper method to check if cleanup is needed
async isCleanupNeeded() {
    const mem = process.memoryUsage();
    const heapUsagePercent = (mem.heapUsed / mem.heapTotal) * 100;
    
    return {
        needed: heapUsagePercent > 70 || this._tempBuffers?.length > 0,
        stats: {
            heapUsagePercent,
            heapUsed: Math.round(mem.heapUsed / 1024 / 1024) + 'MB',
            tempBuffers: this._tempBuffers?.length || 0
        }
    };
}

// Helper method to perform emergency cleanup
async performEmergencyCleanup() {
    logger.warn('Performing emergency cleanup');
    
    try {
        // Reset all potential memory-holding resources
        this.signatureGenerator = null;
        this._tempBuffers = [];
        
        const sharp = require('sharp');
        sharp.cache(false);
        sharp.concurrency(1);
        
        if (global.gc) {
            global.gc();
        }
        
        return true;
    } catch (error) {
        logger.error('Emergency cleanup failed:', error);
        return false;
    }
}

async optimizeMemory() {
    try {
        // Clear Sharp cache aggressively
        sharp.cache(false);
        sharp.concurrency(1);
        
        // Reset internal caches
        if (this.signatureGenerator) {
            await this.signatureGenerator.cleanupTemporaryResources();
        }

        // Clear temporary buffers
        if (this._tempBuffers && this._tempBuffers.length) {
            for (const buffer of this._tempBuffers) {
                if (Buffer.isBuffer(buffer)) {
                    buffer.fill(0);
                }
            }
            this._tempBuffers = [];
        }

        // Force garbage collection
        if (global.gc) {
            await new Promise(resolve => setTimeout(resolve, 100));
            global.gc();
        }

        return true;
    } catch (error) {
        logger.error('Memory optimization failed:', error);
        return false;
    }
}

// Helper method to check if memory optimization is needed
async isMemoryOptimizationNeeded() {
    const mem = process.memoryUsage();
    const heapUsagePercent = (mem.heapUsed / mem.heapTotal) * 100;
    
    return {
        needed: heapUsagePercent > 70,
        stats: {
            heapUsagePercent,
            heapUsed: Math.round(mem.heapUsed / 1024 / 1024) + 'MB',
            heapTotal: Math.round(mem.heapTotal / 1024 / 1024) + 'MB'
        }
    };
}


    async updateProgress(entry, status) {
        try {
            if (status === 'success') {
                this.progress.successful++;
            } else if (status === 'failed') {
                this.progress.failed++;
            }

            if (entry) {
                this.progress.lastProcessedPath = entry.path_lower;
            }
            
            this.progress.processed++;
            this.progress.lastSave = Date.now();

            await fs.writeFile(
                path.join(__dirname, 'migration_progress.json'),
                JSON.stringify(this.progress, null, 2)
            );

            logger.info('Progress updated:', {
                processed: this.progress.processed,
                total: this.progress.total,
                successful: this.progress.successful,
                failed: this.progress.failed,
                percentage: ((this.progress.processed / this.progress.total) * 100).toFixed(2)
            });
        } catch (error) {
            logger.warn('Failed to update progress:', { error: error.message });
        }
    }


    async execute() {
        try {
            const entries = await this.processWithTimeout(
                dropboxManager.fetchDropboxEntries(''),
                this.options.timeout
            );

            const imageFiles = entries.result.entries.filter(entry =>
                ['.jpg', '.jpeg', '.png', '.webp'].some(ext =>
                    entry.path_lower.endsWith(ext)
                )
            );
            
            this.progress.total = imageFiles.length;
            this.progress.startTime = Date.now();

            const concurrency = this.options.memoryConstrained ? 1 : 3;
            const limit = pLimit(concurrency);
            
            for (let i = 0; i < imageFiles.length; i += this.options.batchSize) {
                const batch = imageFiles.slice(i, i + this.options.batchSize);
                
                const results = await Promise.allSettled(
                    batch.map(entry => limit(() => this.processImage(entry)))
                );

                results.forEach((result, index) => {
                    if (result.status === 'fulfilled') {
                        this.progress.successful++;
                    } else {
                        this.progress.failed++;
                        this.progress.errors.push({
                            path: batch[index].path_lower,
                            error: result.reason.message
                        });
                    }
                    this.progress.processed++;
                });

                await this.updateProgress();

                // Memory management between batches
                await this.optimizeMemory();
                
                if (this.options.memoryConstrained) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }

            return {
                success: this.progress.successful,
                failed: this.progress.failed,
                total: this.progress.total
            };

        } catch (error) {
            error.phase = 'execution';
            throw error;
        }
    }
}

module.exports = SignatureMigration;
