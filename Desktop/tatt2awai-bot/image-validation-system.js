// In image-validation-system.js
const fs = require('fs').promises;
const path = require('path');
const logger = require('./logger');

class ImageValidationSystem {
    static STATE_FILE = path.join(__dirname, 'validation-state.json');
    static processedFiles = new Set();
    static validationProgress = {
        lastProcessedFile: null,
        processedCount: 0,
        totalFiles: 0
    };

    static async loadState() {
        try {
            const data = await fs.readFile(this.STATE_FILE, 'utf8');
            const state = JSON.parse(data);
            this.processedFiles = new Set(state.processedFiles);
            this.validationProgress = state.validationProgress;
            logger.info('Loaded validation state:', {
                processedFiles: this.processedFiles.size,
                lastFile: this.validationProgress.lastProcessedFile
            });
        } catch (error) {
            logger.info('No previous validation state found, starting fresh');
            this.processedFiles = new Set();
            this.validationProgress = {
                lastProcessedFile: null,
                processedCount: 0,
                totalFiles: 0
            };
        }
    }

    static async saveState() {
        try {
            const state = {
                processedFiles: Array.from(this.processedFiles),
                validationProgress: this.validationProgress
            };
            await fs.writeFile(this.STATE_FILE, JSON.stringify(state, null, 2));
        } catch (error) {
            logger.error('Failed to save validation state:', error);
        }
    }

    static async validateAllImages(dropboxManager) {
        // Load existing state first
        await this.loadState();

        logger.info('Starting comprehensive image validation...');
        const results = {
            total: 0,
            valid: 0,
            invalid: [],
            warnings: []
        };

        try {
            const allImages = await dropboxManager.fetchDropboxEntries('');
            if (!allImages?.result?.entries) {
                throw new Error('Failed to fetch images from Dropbox');
            }

            results.total = allImages.result.entries.length;
            this.validationProgress.totalFiles = results.total;

            // Filter out already processed files
            const remainingImages = allImages.result.entries.filter(img => 
                !this.processedFiles.has(img.path_lower)
            );

            logger.info(`Resuming validation: ${this.processedFiles.size} files already processed, ${remainingImages.length} remaining`);

            // Process in smaller batches
            const batchSize = 50;
            for (let i = 0; i < remainingImages.length; i += batchSize) {
                const batch = remainingImages.slice(i, i + batchSize);
                
                await Promise.all(batch.map(async (image) => {
                    try {
                        const validationResult = await this.validateSingleImage(image, dropboxManager);
                        
                        if (validationResult.isValid) {
                            results.valid++;
                        } else {
                            results.invalid.push({
                                path: image.path_lower,
                                reason: validationResult.reason
                            });
                        }

                        if (validationResult.warnings.length > 0) {
                            results.warnings.push({
                                path: image.path_lower,
                                warnings: validationResult.warnings
                            });
                        }

                        // Track processed files
                        this.processedFiles.add(image.path_lower);
                        this.validationProgress.lastProcessedFile = image.path_lower;
                        this.validationProgress.processedCount = this.processedFiles.size;

                        // Save state after each file
                        if (this.processedFiles.size % 10 === 0) { // Save every 10 files
                            await this.saveState();
                        }

                    } catch (error) {
                        results.invalid.push({
                            path: image.path_lower,
                            reason: `Validation error: ${error.message}`
                        });
                    }
                }));

                // Log progress and save state after each batch
                const totalProcessed = this.processedFiles.size;
                logger.info(`Validated ${totalProcessed}/${results.total} images`);
                await this.saveState();
            }

            await this.saveState(); // Final save
            return results;

        } catch (error) {
            logger.error('Validation failed:', error);
            await this.saveState(); // Save on error too
            throw error;
        }
    }

    static async validateSingleImage(image, dropboxManager) {
        const result = {
            isValid: false,
            reason: null,
            warnings: []
        };

        try {
            // Check file extension
            if (!image.path_lower.match(/\.(jpg|jpeg|png|webp)$/i)) {
                result.reason = 'Unsupported file format';
                return result;
            }

            // Download and validate image
            const fileData = await dropboxManager.downloadFile(image.path_lower);
            if (!fileData?.result?.fileBinary) {
                result.reason = 'Failed to download file';
                return result;
            }

            const buffer = Buffer.isBuffer(fileData.result.fileBinary) ?
                fileData.result.fileBinary :
                Buffer.from(fileData.result.fileBinary);

            // Validate image integrity
            const metadata = await sharp(buffer).metadata();
            
            // Check dimensions
            if (metadata.width < 100 || metadata.height < 100) {
                result.reason = 'Image dimensions too small';
                return result;
            }

            if (metadata.width > 10000 || metadata.height > 10000) {
                result.reason = 'Image dimensions too large';
                return result;
            }

            // Check for corruption
            try {
                await sharp(buffer)
                    .resize(32, 32)
                    .toBuffer();
            } catch (error) {
                result.reason = 'Image data corrupted';
                return result;
            }

            // Add warnings for potential issues
            if (metadata.width < 300 || metadata.height < 300) {
                result.warnings.push('Low resolution image');
            }

            if (buffer.length > 10 * 1024 * 1024) { // 10MB
                result.warnings.push('Large file size');
            }

            // Validate color space
            if (metadata.space !== 'srgb') {
                result.warnings.push('Non-standard color space');
            }

            // Check for extreme aspect ratios
            const aspectRatio = metadata.width / metadata.height;
            if (aspectRatio > 3 || aspectRatio < 0.33) {
                result.warnings.push('Extreme aspect ratio');
            }

            // All checks passed
            result.isValid = true;
            return result;

        } catch (error) {
            result.reason = `Validation error: ${error.message}`;
            return result;
        }
    }
}

module.exports = ImageValidationSystem;
