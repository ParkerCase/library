const fs = require('fs');
const path = require('path');
const vision = require('@google-cloud/vision');
const sharp = require('sharp');
const crypto = require('crypto');
const logger = require('./logger');
const { createClient } = require('@supabase/supabase-js');
const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_KEY
);


// Constants
// Constants
const SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
const CACHE_SETTINGS = {
    DURATION: 24 * 60 * 60 * 1000, // 24 hours
    ANALYSIS_DURATION: 24 * 60 * 60 * 1000 // 24 hours
};

const IMAGE_PROCESSING = {
    MAX_SIZE: 4096,
    QUALITY: 90,
    MIN_SIZE: 300,
    OPTIMAL_DPI: 300
};

const SEQUENCE_PATTERNS = [
    /before|after|during/i,
    /removal/i,
    /t[0-9]+/i,  // T1, T2, etc.
    /session[0-9]+/i,
    /treatment[0-9]+/i,
    /\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b/ // dates
];

const ANALYSIS_THRESHOLDS = {
    TATTOO_CONFIDENCE: 0.7,
    COLOR_MATCH: 0.85,
    SEQUENCE_SIMILARITY: 0.80,
    MIN_SEQUENCE_IMAGES: 2,
    MAX_SEQUENCE_GAP_DAYS: 90,
    SKIN_REDNESS_THRESHOLD: 0.6,
    EDGE_SHARPNESS: 0.65,
    MIN_COLOR_PROMINENCE: 0.05
};

// Initialize Vision Client
const visionClient = new vision.ImageAnnotatorClient({
    keyFilename: path.join(__dirname, process.env.GOOGLE_APPLICATION_CREDENTIALS),
});

// Cache for analysis results
const analysisCache = new Map();


// Vision API credential testing
async function testVisionCredentials() {
    try {
        const testImage = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', 'base64');
        await visionClient.labelDetection({image: {content: testImage}});
        logger.info('Vision API credentials verified successfully');
        return true;
    } catch (error) {
        logger.error('Vision API credential test failed:', {
            error: error.message,
            stack: error.stack
        });
        return false;
    }
}

// Color Analysis Utilities
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
    },

    identifyDominantColors(colors, threshold = ANALYSIS_THRESHOLDS.MIN_COLOR_PROMINENCE) {
        return colors
            .filter(color => color.pixelFraction >= threshold)
            .sort((a, b) => b.pixelFraction - a.pixelFraction);
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

// Core ImageProcessor Implementation
const imageProcessor = {
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

async processImage(file) {
    let optimizedBuffer = null;
    let tempPath = null;
    let processingStage = 'validation';
    let startTime = Date.now();

    try {
        // Input validation
        if (!file) {
            throw new Error('No file provided');
        }

        // Handle buffer input
        if (file.buffer) {
            tempPath = path.join('uploads', `temp_${Date.now()}_${path.basename(file.path || 'temp.jpg')}`);
            fs.writeFileSync(tempPath, file.buffer);
            file.path = tempPath;
        }

        // Generate cache key early
        const cacheKey = await this.generateFileHash(file.path);
        
        // Check Supabase cache first
        const { data: cachedAnalysis } = await supabase
            .from('image_analysis')
            .select('*')
            .eq('path', file.path)
            .single();

        if (cachedAnalysis && 
            (Date.now() - new Date(cachedAnalysis.analyzed_at).getTime()) < CACHE_SETTINGS.DURATION) {
            logger.info('Returning cached analysis from Supabase', { path: file.path });
            return cachedAnalysis.analysis;
        }

        // Optimize the image
        processingStage = 'optimization';
        optimizedBuffer = await sharp(file.path)
            .resize(2048, 2048, {
                fit: 'inside',
                withoutEnlargement: true,
                kernel: sharp.kernel.lanczos3
            })
            .normalise()
            .toFormat('jpeg', { quality: 90 })
            .toBuffer();

        logger.info('Image optimization completed', {
            path: file.path,
            originalSize: fs.statSync(file.path).size,
            optimizedSize: optimizedBuffer.length
        });

        // Vision API analysis
processingStage = 'vision-api';
const [visionResult] = await visionClient.annotateImage({
    image: { content: optimizedBuffer.toString('base64') },
    features: [
        { type: 'LABEL_DETECTION', maxResults: 20 },
        { type: 'IMAGE_PROPERTIES' },
        { type: 'OBJECT_LOCALIZATION' },
        { type: 'TEXT_DETECTION' },
        { type: 'FACE_DETECTION' }
    ]
});
logger.info('Vision API analysis completed:', {
path: file.path,
    labelsFound: visionResult.labelAnnotations?.length || 0,
    hasObjects: !!visionResult.localizedObjectAnnotations?.length,
    hasColors: !!visionResult.imagePropertiesAnnotation?.dominantColors
});

// Feature analysis
processingStage = 'feature-analysis';
const tattooFeatures = await this.analyzeTattooFeatures(visionResult);
logger.info('Tattoo analysis completed:', {
path: file.path,
    isTattoo: !!tattooFeatures,
    confidence: tattooFeatures?.confidence || 0,
    placement: tattooFeatures?.placement || 'unknown'
});

const colorAnalysisResult = await colorAnalysis.analyzePalette(
    visionResult.imagePropertiesAnnotation?.dominantColors?.colors || []
);
logger.info('Color analysis completed:', {
    path: file.path,
    colorsFound: colorAnalysisResult?.length || 0,
    dominantColors: colorAnalysisResult?.slice(0, 3)?.map(c => ({
        rgb: c.rgb,
        prominence: c.prominence
    }))
});


// Quality assessment
const qualityMetrics = await qualityAssessment.assessQuality(optimizedBuffer);
logger.info('Quality assessment completed:', {
    path: file.path,
    dimensions: qualityMetrics.dimensions,
    hasQualityMetrics: !!qualityMetrics.statistics,
    stage: processingStage
});
        // Compile analysis results
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

        // Store in cache
        analysisCache.set(cacheKey, {
            data: analysis,
            timestamp: Date.now(),
            processingTime: Date.now() - startTime
        });

        // Store in Supabase
        await supabase
            .from('image_analysis')
            .upsert({
                path: file.path,
                analysis: analysis,
                analyzed_at: new Date().toISOString()
            });

        logger.info('Analysis completed successfully', {
            path: file.path,
            hasFeatures: !!analysis.tattooFeatures,
            processingTime: Date.now() - startTime
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
},
    async analyzeTattooFeatures(visionResult) {
try {
        // More comprehensive tattoo keywords
        const tattooKeywords = [
            'tattoo', 'ink', 'body art', 'skin art', 'tribal',
            'design', 'artwork', 'skin', 'marking', 'permanent'
        ];

        logger.info('Analyzing tattoo features:', {
            labelsFound: visionResult.labelAnnotations?.length || 0
        });

        const relevantLabels = visionResult.labelAnnotations?.filter(label => 
            tattooKeywords.some(t => label.description.toLowerCase().includes(t))
        ) || [];

        // Lower threshold for tattoo detection
        const isTattoo = relevantLabels.length > 0 || 
                        this.hasBodyPartContext(visionResult.labelAnnotations) ||
                        this.hasTattooColorPatterns(visionResult.imagePropertiesAnnotation);

        if (isTattoo) {
            const features = {
                isTattoo: true,
                confidence: Math.max(...relevantLabels.map(l => l.score || 0.5)),
                inkColors: [],
                placement: this.determineBodyPlacement(visionResult.labelAnnotations),
                characteristics: this.analyzeCharacteristics(visionResult),
                inkDensity: this.calculateInkDensity(visionResult.imagePropertiesAnnotation),
                fadingMetrics: this.calculateFadingMetrics(visionResult)
            };

            logger.info('Tattoo features detected:', {
                confidence: features.confidence,
                placement: features.placement,
                hasInkColors: features.inkColors.length > 0
            });

            return features;
        }

        return null;
    } catch (error) {
        logger.error('Error analyzing tattoo features:', error);
        return null;
    }
},

    calculateProgress(firstImage, currentImage) {
        return {
            densityReduction: this.calculateInkDensity(firstImage, currentImage),
            patternChanges: this.detectPatternChanges(firstImage, currentImage),
            fadingMetrics: {
                overallFading: this.calculateOverallFading(),
                uniformity: this.calculateFadingUniformity(),
                areas: this.detectFadedAreas()
            },
            timeElapsed: new Date(currentImage.metadata.modified) - new Date(firstImage.metadata.modified)
        };
    },

    calculateInkDensity(firstImage, currentImage) {
        // Implementation
        return this.calculateDensityFromStats(firstImage.stats, currentImage.stats);
    },

    detectPatternChanges(firstImage, currentImage) {
        // Implementation
        return {
            fragmentation: this.detectFragmentation(firstImage, currentImage),
            dotting: this.detectDotting(firstImage, currentImage)
        };
    },

    detectVisualProgression(sequence) {
        return {
            original: this.detectOriginalTattoo(sequence[0]),
            fragmentation: this.detectFragmentation(sequence),
            fading: this.detectFading(sequence),
            clearing: this.detectClearing(sequence)
        };
    },

    // Helper methods
    detectOriginalTattoo(image) {
        return {
            isOriginal: true, // Implement logic
            confidence: 0.9,  // Implement logic
            features: {}      // Implement logic
        };
    },

    detectFragmentation(sequence) {
        return {
            hasFragmentation: true, // Implement logic
            pattern: 'dotted',      // Implement logic
            coverage: 0.8           // Implement logic
        };
    },

    detectFading(sequence) {
        return {
            fadingLevel: 'moderate',  // Implement logic
            uniformity: 0.7,          // Implement logic
            areas: []                 // Implement logic
        };
    },

    detectClearing(sequence) {
        return {
            clearanceLevel: 'partial',  // Implement logic
            progress: 0.6,              // Implement logic
            remainingAreas: []          // Implement logic
        };
    },

    calculateDensityFromStats(stats1, stats2) {
        // Implement density calculation
        return 0.7; // Placeholder
    }
},


    analyzeCharacteristics(visionResult) {
        return {
            style: this.determineStyle(visionResult),
            complexity: this.assessComplexity(visionResult),
            edges: this.analyzeEdges(visionResult)
        };
    },

    determineStyle(visionResult) {
        // Style determination logic
        return 'traditional'; // Placeholder
    },

    assessComplexity(visionResult) {
        // Complexity assessment logic
        return 0.5; // Placeholder
    },

    analyzeEdges(visionResult) {
        // Edge analysis logic
        return { sharpness: 0.7 }; // Placeholder
    },
hasBodyPartContext(labels) {
    const bodyParts = ['skin', 'arm', 'leg', 'back', 'chest', 'shoulder', 'body'];
    return labels?.some(label => 
        bodyParts.some(part => label.description.toLowerCase().includes(part))
    );
},

hasTattooColorPatterns(imageProperties) {
    if (!imageProperties?.dominantColors?.colors) return false;
    
    const colors = imageProperties.dominantColors.colors;
    const hasHighContrast = this.calculateColorContrast(colors) > 0.4;
    const hasInkLikeColors = colors.some(color => 
        this.isInkLikeColor(color.color)
    );
    
    return hasHighContrast || hasInkLikeColors;
},

calculateColorContrast(colors) {
    if (colors.length < 2) return 0;
    const brightnesses = colors.map(color => 
        (color.color.red + color.color.green + color.color.blue) / (3 * 255)
    );
    return Math.max(...brightnesses) - Math.min(...brightnesses);
},
isInkLikeColor(color) {
    const brightness = (color.red + color.green + color.blue) / (3 * 255);
    return brightness < 0.5; // Dark colors are more likely to be ink
},

    async generateFileHash(filePath) {
        const content = fs.readFileSync(filePath);
        return crypto.createHash('sha256').update(content).digest('hex');
    },

    async optimizeImage(filePath) {
        try {
            const image = sharp(filePath);
            const metadata = await image.metadata();

            return image
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
        } catch (error) {
            logger.error('Error optimizing image:', error);
            throw error;
        }
    },

    determineBodyPlacement(labels) {
        const bodyParts = {
            arm: ['arm', 'forearm', 'bicep', 'shoulder'],
            leg: ['leg', 'thigh', 'calf', 'ankle'],
            torso: ['chest', 'back', 'stomach', 'abdomen'],
            head: ['neck', 'face', 'head', 'scalp']
        };

        for (const [area, parts] of Object.entries(bodyParts)) {
            if (labels?.some(label => 
                parts.some(part => label.description.toLowerCase().includes(part))
            )) {
                return area;
            }
        }
        return 'unknown';
    }

};

// Export the imageProcessor with all its methods bound to the correct context
module.exports = {
    processImage: imageProcessor.processImage.bind(imageProcessor),
    getFileType: imageProcessor.getFileType.bind(imageProcessor),
    isImage: imageProcessor.isImage.bind(imageProcessor),
    analyzeTattooFeatures: imageProcessor.analyzeTattooFeatures.bind(imageProcessor),
    testVisionCredentials,
    SUPPORTED_IMAGE_TYPES,
    validateImageFile: imageProcessor.validateImageFile.bind(imageProcessor),
    // Add these new exports
    calculateProgress: imageProcessor.calculateProgress.bind(imageProcessor),
    detectVisualProgression: imageProcessor.detectVisualProgression.bind(imageProcessor),
    detectPatternChanges: imageProcessor.detectPatternChanges.bind(imageProcessor)
};
