const sharp = require('sharp');
const path = require('path');
const logger = require('./logger');

class EnhancedValidation {
    // Image content checks
    static async validateImageContent(buffer) {
        const checks = {
            isValidImage: false,
            hasMinimumQuality: false,
            hasAppropriateSize: false,
            isProcessable: false,
            details: {
                dimensions: null,
                format: null,
                quality: null
            },
            warnings: []
        };

        try {
            // Basic image validation
            const image = sharp(buffer);
            const metadata = await image.metadata();
            const stats = await image.stats();

            checks.details.dimensions = {
                width: metadata.width,
                height: metadata.height,
                aspectRatio: metadata.width / metadata.height
            };
            checks.details.format = metadata.format;

            // Size validation
            const isValidSize = metadata.width >= 100 && 
                              metadata.height >= 100 && 
                              metadata.width <= 10000 && 
                              metadata.height <= 10000;
            checks.hasAppropriateSize = isValidSize;

            if (!isValidSize) {
                checks.warnings.push(`Invalid image dimensions: ${metadata.width}x${metadata.height}`);
            }

            // Quality checks
            const { quality, warnings } = await this.assessImageQuality(buffer);
            checks.details.quality = quality;
            checks.hasMinimumQuality = quality.overall >= 0.5;
            checks.warnings.push(...warnings);

            // Processability check
            try {
                await this.testImageProcessing(buffer);
                checks.isProcessable = true;
            } catch (error) {
                checks.warnings.push(`Processing test failed: ${error.message}`);
                checks.isProcessable = false;
            }

            checks.isValidImage = checks.isProcessable && 
                                checks.hasAppropriateSize && 
                                checks.hasMinimumQuality;

            return checks;

        } catch (error) {
            logger.error('Image validation failed:', error);
            return {
                ...checks,
                error: error.message
            };
        }
    }

    // Quality assessment
    static async assessImageQuality(buffer) {
        const warnings = [];
        const image = sharp(buffer);
        const stats = await image.stats();
        const metadata = await image.metadata();

        // Calculate various quality metrics
        const contrast = this.calculateContrast(stats);
        const sharpness = await this.estimateSharpness(buffer);
        const noise = await this.estimateNoise(buffer);
        const exposure = this.calculateExposure(stats);

        // Quality scores (0-1)
        const quality = {
            contrast: contrast,
            sharpness: sharpness,
            noise: 1 - noise,
            exposure: exposure,
            overall: (contrast + sharpness + (1 - noise) + exposure) / 4
        };

        // Generate warnings based on quality
        if (contrast < 0.3) warnings.push('Low contrast');
        if (sharpness < 0.3) warnings.push('Image appears blurry');
        if (noise > 0.7) warnings.push('High noise levels');
        if (exposure < 0.3) warnings.push('Poor exposure');

        return { quality, warnings };
    }

    static async testImageProcessing(buffer) {
        // Test basic image operations
        const image = sharp(buffer);
        
        await Promise.all([
            // Test resize
            image.clone().resize(100, 100).toBuffer(),
            // Test format conversion
            image.clone().jpeg().toBuffer(),
            // Test color manipulation
            image.clone().greyscale().toBuffer(),
            // Test edge detection
            image.clone().convolve({
                width: 3,
                height: 3,
                kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
            }).toBuffer()
        ]);
    }

    // Helper methods
    static calculateContrast(stats) {
        return stats.channels.reduce((acc, channel) => {
            const range = channel.max - channel.min;
            return acc + (range / 255);
        }, 0) / stats.channels.length;
    }

    static async estimateSharpness(buffer) {
        const edge = await sharp(buffer)
            .greyscale()
            .convolve({
                width: 3,
                height: 3,
                kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
            })
            .raw()
            .toBuffer();

        const edgeStrength = Array.from(edge).reduce((sum, val) => sum + val, 0) / edge.length;
        return Math.min(edgeStrength / 30, 1);
    }

    static async estimateNoise(buffer) {
        const data = await sharp(buffer)
            .greyscale()
            .raw()
            .toBuffer();

        let noiseEstimate = 0;
        const width = Math.sqrt(data.length);
        
        for (let i = 1; i < data.length - 1; i++) {
            const diff = Math.abs(data[i] - data[i - 1]);
            noiseEstimate += diff;
        }

        return noiseEstimate / (data.length * 255);
    }

    static calculateExposure(stats) {
        const means = stats.channels.map(c => c.mean / 255);
        const avgExposure = means.reduce((a, b) => a + b) / means.length;
        return 1 - Math.abs(0.5 - avgExposure) * 2;
    }
}

module.exports = EnhancedValidation;
