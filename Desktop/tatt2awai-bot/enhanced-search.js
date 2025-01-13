const EnhancedSignatureStore = require('./enhanced-store');
const dropboxManager = require('./dropboxManager');
const logger = require('./logger');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs').promises;

class EnhancedVisualSearch {
    constructor() {
        this.isInitialized = false;
        this.initializePromise = null;
    }

    async initialize() {
        if (this.initializePromise) return this.initializePromise;

        this.initializePromise = (async () => {
            try {
                logger.info('Initializing visual search system...');
                await EnhancedSignatureStore.initialize(dropboxManager);
                this.isInitialized = true;
                logger.info('Visual search system initialized successfully');
            } catch (error) {
                logger.error('Failed to initialize visual search:', error);
                throw error;
            }
        })();

        return this.initializePromise;
    }

    async search(imageData, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Visual search system not initialized');
        }

        const startTime = Date.now();
        const metrics = {
            preprocessing: 0,
            search: 0,
            sequenceAnalysis: 0,
            total: 0
        };

        try {
            // Ensure we have valid image data
            const buffer = await this.preprocessImage(imageData);
            metrics.preprocessing = Date.now() - startTime;

            // Get all images from Dropbox
            const allImages = await dropboxManager.fetchDropboxEntries('');
            if (!allImages?.result?.entries) {
                throw new Error('Failed to fetch images from Dropbox');
            }

            // Find exact match
            const searchStart = Date.now();
            const match = await EnhancedSignatureStore.findExactMatch(buffer, allImages);
            metrics.search = Date.now() - searchStart;

            if (!match) {
                return {
                    success: false,
                    message: 'No matching image found',
                    metrics
                };
            }

            // If exact match found, find sequence
            const sequenceStart = Date.now();
            const sequence = match.isExact ?
                await EnhancedSignatureStore.findSequence(match, allImages) :
                null;
            metrics.sequenceAnalysis = Date.now() - sequenceStart;

            metrics.total = Date.now() - startTime;

            return {
                success: true,
                match: {
                    path: match.path,
                    confidence: match.confidence,
                    isExact: match.isExact,
                    score: match.score,
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
                metrics: {
                    ...metrics,
                    timings: {
                        preprocessing: `${metrics.preprocessing}ms`,
                        search: `${metrics.search}ms`,
                        sequenceAnalysis: `${metrics.sequenceAnalysis}ms`,
                        total: `${metrics.total}ms`
                    }
                }
            };

        } catch (error) {
            logger.error('Search failed:', error);
            throw error;
        }
    }

    async preprocessImage(imageData) {
        try {
            // Handle different input types
            let buffer;
            if (Buffer.isBuffer(imageData)) {
                buffer = imageData;
            } else if (imageData.buffer) {
                buffer = imageData.buffer;
            } else if (imageData.path) {
                buffer = await fs.readFile(imageData.path);
            } else {
                throw new Error('Invalid image data provided');
            }

            // Normalize image
            return await sharp(buffer)
                .resize(1000, 1000, {
                    fit: 'inside',
                    withoutEnlargement: true
                })
                .normalize()
                .toBuffer();

        } catch (error) {
            logger.error('Image preprocessing failed:', error);
            throw error;
        }
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            storeStatus: EnhancedSignatureStore.getStatus()
        };
    }
}

module.exports = new EnhancedVisualSearch();
