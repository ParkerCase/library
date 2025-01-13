const sharp = require('sharp');
const path = require('path');

// Optimization utilities for image processing pipeline
class ImageOptimizer {
    static async optimizeForSignature(buffer) {
        // Pre-process image for optimal signature generation
        return sharp(buffer)
            .resize(1000, 1000, {
                fit: 'inside',
                withoutEnlargement: true,
                kernel: sharp.kernel.lanczos3
            })
            .normalize()
            .clahe({ // Contrast Limited Adaptive Histogram Equalization
                width: 8,
                height: 8,
                maxSlope: 3
            })
            .toBuffer();
    }

    static async generateThumbnails(buffer) {
        // Generate multiple scales in parallel
        const [tiny, small, medium] = await Promise.all([
            sharp(buffer).resize(16, 16, { fit: 'fill' }).raw().toBuffer(),
            sharp(buffer).resize(32, 32, { fit: 'fill' }).raw().toBuffer(),
            sharp(buffer).resize(64, 64, { fit: 'fill' }).raw().toBuffer()
        ]);

        return { tiny, small, medium };
    }

    static async detectBlurAndExposure(buffer) {
        const metadata = await sharp(buffer).metadata();
        const stats = await sharp(buffer).stats();

        // Calculate blur score using Laplacian variance
        const laplacian = await sharp(buffer)
            .greyscale()
            .convolve({
                width: 3,
                height: 3,
                kernel: [0, 1, 0, 1, -4, 1, 0, 1, 0]
            })
            .raw()
            .toBuffer();

        const variance = this.calculateVariance(laplacian);
        const exposureScore = this.calculateExposureScore(stats);

        return {
            isBlurry: variance < 500,
            blurScore: variance,
            exposureScore,
            needsEnhancement: variance < 500 || exposureScore < 0.4
        };
    }

    static calculateVariance(buffer) {
        const pixels = new Uint8Array(buffer);
        const mean = pixels.reduce((sum, val) => sum + val, 0) / pixels.length;
        return pixels.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / pixels.length;
    }

    static calculateExposureScore(stats) {
        const meanLuminance = stats.channels.reduce((sum, channel) => sum + channel.mean, 0) / 
            (stats.channels.length * 255);
        return 1 - Math.abs(0.5 - meanLuminance) * 2;
    }

    static async enhanceImage(buffer) {
        const { isBlurry, exposureScore } = await this.detectBlurAndExposure(buffer);
        let enhanced = sharp(buffer);

        if (isBlurry) {
            enhanced = enhanced.sharpen({
                sigma: 1.5,
                m1: 1.5,
                m2: 2.0
            });
        }

        if (exposureScore < 0.4) {
            enhanced = enhanced.modulate({
                brightness: 1.1,
                saturation: 1.2
            });
        }

        return enhanced.normalize().toBuffer();
    }

    static async validateImage(buffer) {
        try {
            const metadata = await sharp(buffer).metadata();
            const minSize = 100; // Minimum dimension size
            const maxSize = 10000; // Maximum dimension size

            return {
                isValid: metadata.width >= minSize && 
                        metadata.height >= minSize &&
                        metadata.width <= maxSize &&
                        metadata.height <= maxSize,
                metadata,
                errorCode: null
            };
        } catch (error) {
            return {
                isValid: false,
                metadata: null,
                errorCode: error.code || 'UNKNOWN_ERROR'
            };
        }
    }
}

// Memory management utilities
class MemoryManager {
    static async optimizeMemoryUsage() {
        if (process.memoryUsage().heapUsed > 500 * 1024 * 1024) { // 500MB threshold
            if (global.gc) {
                global.gc();
            }
            return new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    static createMemoryEfficientCache(maxSize = 1000) {
        return new Map();
    }

    static monitorMemoryUsage(threshold = 0.8) { // 80% threshold
        const used = process.memoryUsage().heapUsed;
        const total = process.memoryUsage().heapTotal;
        
        if (used / total > threshold) {
            console.warn('High memory usage detected:', {
                used: `${Math.round(used / 1024 / 1024)}MB`,
                total: `${Math.round(total / 1024 / 1024)}MB`,
                percentage: `${Math.round(used / total * 100)}%`
            });
        }
    }
}

// Batch processing utilities
class BatchProcessor {
    static async processBatch(items, processor, options = {}) {
        const {
            batchSize = 50,
            concurrency = 5,
            retryAttempts = 3,
            retryDelay = 1000
        } = options;

        const results = [];
        const errors = [];

        for (let i = 0; i < items.length; i += batchSize) {
            const batch = items.slice(i, i + batchSize);
            const batchPromises = batch.map(item => 
                this.processWithRetry(item, processor, retryAttempts, retryDelay)
            );

            const batchResults = await Promise.allSettled(batchPromises);
            
            batchResults.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    results.push(result.value);
                } else {
                    errors.push({
                        item: batch[index],
                        error: result.reason
                    });
                }
            });

            // Memory management between batches
            await MemoryManager.optimizeMemoryUsage();
            
            // Optional delay between batches
            if (i + batchSize < items.length) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        return { results, errors };
    }

    static async processWithRetry(item, processor, attempts, delay) {
        for (let attempt = 0; attempt < attempts; attempt++) {
            try {
                return await processor(item);
            } catch (error) {
                if (attempt === attempts - 1) throw error;
                await new Promise(resolve => 
                    setTimeout(resolve, delay * Math.pow(2, attempt))
                );
            }
        }
    }
}

module.exports = {
    ImageOptimizer,
    MemoryManager,
    BatchProcessor
};
