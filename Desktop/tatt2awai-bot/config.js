// config.js
module.exports = {
    // System-wide settings
    system: {
        maxThreads: 4,
        maxMemory: '2GB',
        tempDirectory: '/tmp/signatures',
        cleanupInterval: 3600000 // 1 hour
    },

    // Feature extraction settings
    features: {
        color: {
            dominantColorCount: 8,
            quantizationLevels: {
                rgb: 64,
                lab: 32,
                hsv: 48
            },
            histogramSmoothing: 0.1
        },
        edge: {
            cannyLow: 0.1,
            cannyHigh: 0.3,
            orientationBins: 36,
            minEdgeLength: 5
        },
        texture: {
            gaborOrientations: [0, 30, 60, 90, 120, 150],
            gaborFrequencies: [0.1, 0.2, 0.3, 0.4],
            lbpRadius: 2,
            glcmDistances: [1, 2, 4]
        },
        hash: {
            dctSize: 32,
            hashSize: 256,
            errorCorrectionLevel: 'medium'
        }
    },

    // Performance optimization settings
    optimization: {
        caching: {
            enabled: true,
            maxSize: '1GB',
            ttl: 3600
        },
        processing: {
            batchSize: 100,
            timeout: 30000,
            retryAttempts: 3
        }
    },

    // Quality thresholds
    quality: {
        minConfidence: 0.8,
        minReliability: 0.7,
        maxNoiseLevel: 0.2
    }
};
