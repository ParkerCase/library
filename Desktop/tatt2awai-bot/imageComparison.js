const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const logger = require('./logger');
const { ImageSignatureGenerator } = require('./imageProcessor');
const dropboxManager = require('./dropboxManager');
const os = require('os');
const { v4: uuidv4 } = require('uuid');

// Constants for image comparison

const ADVANCED_SETTINGS = {
    HASH_SIZES: [16, 32, 64, 128],
    ROTATIONS: [0, 90, 180, 270],
    SCALES: [0.5, 1.0, 2.0],
    COLOR_SPACES: ['srgb', 'lab', 'hsv']
};

const COMPARISON_THRESHOLDS = {
    EXACT_MATCH: 0.98,
    HIGH_CONFIDENCE: 0.95,
    MEDIUM_CONFIDENCE: 0.85,
    LOW_CONFIDENCE: 0.70,
    MIN_VALIDATION_SCORE: 0.90,
    SEQUENCE_SIMILARITY: 0.85,
    COLOR_TOLERANCE: 0.1,
    ASPECT_RATIO_TOLERANCE: 0.05
};

const COMPARISON_WEIGHTS = {
    PERCEPTUAL_HASH: 0.25,
    COLOR_METRICS: 0.25,
    EDGE_SIGNATURE: 0.20,
    STRUCTURAL: 0.15,
    METADATA: 0.15
};

const QUALITY_THRESHOLDS = {
    SHARPNESS: 0.5,
    CONTRAST: 0.4,
    SNR: 0.3
};

const FEATURE_MATCHING = {
    MIN_MATCHES: 10,
    DISTANCE_THRESHOLD: 0.8,
    RANSAC_THRESHOLD: 3.0
};

class ImageComparison {
 static async preprocessImage(buffer) {
        try {
            const image = sharp(buffer);
            
            const normalized = await image
                .normalize()
                .gamma(1.2)
                .removeAlpha()
                .toBuffer();

            const denoised = await sharp(normalized)
                .median(3)
                .toBuffer();

            const enhanced = await sharp(denoised)
                .linear(1.1, -0.1)
                .modulate({
                    brightness: 1.05,
                    saturation: 1.1
                })
                .toBuffer();

            return {
                original: buffer,
                normalized: normalized,
                denoised: denoised,
                enhanced: enhanced
            };
        } catch (error) {
            logger.error('Preprocessing failed:', error);
            throw error;
        }
    }

// Add after preprocessImage method
    static async generateAdvancedHash(buffer) {
        const hashPromises = ADVANCED_SETTINGS.HASH_SIZES.flatMap(size => 
            ADVANCED_SETTINGS.ROTATIONS.map(async rotation => {
                const rotated = await sharp(buffer)
                    .rotate(rotation)
                    .toBuffer();
                    
                return new Promise((resolve, reject) => {
                    imageHash(rotated, size, true, (error, hash) => {
                        if (error) reject(error);
                        else resolve({
                            size,
                            rotation,
                            hash,
                            wavelet: this.generateWaveletHash(rotated)
                        });
                    });
                });
            })
        );

        return await Promise.all(hashPromises);
    }


// Replace the existing compareImages method (around line 50-150)
    static async compareImages(searchImage, candidateImage, options = {}) {
        const startTime = Date.now();

        try {
            // Parallel processing of both images
            const [
                searchPreprocess,
                candidatePreprocess,
                searchFeatures,
                candidateFeatures,
                searchPatterns,
                candidatePatterns
            ] = await Promise.all([
                this.preprocessImage(searchImage),
                this.preprocessImage(candidateImage),
                this.extractFeatures(searchImage),
                this.extractFeatures(candidateImage),
                this.analyzePatterns(searchImage),
                this.analyzePatterns(candidateImage)
            ]);

            const comparisons = {
                perceptual: await this.comparePerceptuallyEnhanced(
                    searchPreprocess.enhanced,
                    candidatePreprocess.enhanced
                ),
                structural: this.compareStructuralFeatures(
                    searchFeatures,
                    candidateFeatures
                ),
                semantic: await this.compareSemanticContent(
                    searchPatterns,
                    candidatePatterns
                ),
                contextual: this.compareContextualFeatures(
                    searchPatterns.deepFeatures,
                    candidatePatterns.deepFeatures
                )
            };

            const threshold = this.calculateAdaptiveThreshold(comparisons);
            const confidence = this.calculateConfidenceScore(comparisons, threshold);

            return {
                isMatch: confidence.score >= threshold,
                confidence: confidence.score,
                matchType: this.determineMatchType(confidence),
                details: {
                    comparisons,
                    threshold,
                    processingTime: Date.now() - startTime,
                    qualityMetrics: {
                        search: this.assessImageQuality(searchPreprocess),
                        candidate: this.assessImageQuality(candidatePreprocess)
                    }
                }
            };
        } catch (error) {
            logger.error('Enhanced image comparison failed:', error);
            throw error;
        }
    }

static calculateImageQuality(buffer) {
    return sharp(buffer)
        .stats()
        .then(stats => ({
            sharpness: this.estimateSharpness(stats),
            contrast: this.calculateContrast(stats),
            snr: this.calculateSNR(stats)
        }));
}

static async comparePatterns(patterns1, patterns2) {
    const edgeScore = Math.abs(patterns1.edges - patterns2.edges);
    const textureScore = Math.abs(patterns1.texture - patterns2.texture);
    const gradientScore = Math.abs(patterns1.gradients - patterns2.gradients);
    
    return 1 - ((edgeScore + textureScore + gradientScore) / 3);
}

static async compareSpatialRelations(img1, img2) {
    // Basic spatial comparison
    return 0.8; // Default implementation
}

static async compareSemanticContext(img1, img2) {
    // Basic semantic comparison
    return 0.8; // Default implementation
}

static async performFeatureMatching(img1, img2) {
    const normalizedImg1 = await this.normalizeImage(img1);
    const normalizedImg2 = await this.normalizeImage(img2);
    
    const features1 = await this.extractLocalFeatures(normalizedImg1);
    const features2 = await this.extractLocalFeatures(normalizedImg2);
    
    const matches = await this.matchFeatures(features1, features2);
    
    return {
        matchCount: matches.length,
        confidence: matches.length / Math.min(features1.length, features2.length),
        inlierRatio: matches.filter(m => m.distance < 0.8).length / matches.length
    };
}

static async compareImageContent(features1, features2) {
    try {
        if (!features1 || !features2) {
            return {
                score: 0,
                confidence: 0,
                details: {
                    textureMatch: 0,
                    edgeMatch: 0,
                    shapeMatch: 0,
                    patternMatch: 0,
                    semanticMatch: 0
                }
            };
        }

        // Calculate individual feature scores
        const textureScore = await this.compareTextureProfiles(
            features1.textureProfile,
            features2.textureProfile
        );

        const edgeScore = await this.compareEdgeProfiles(
            features1.edgeProfile,
            features2.edgeProfile
        );

        const shapeScore = this.compareShapeProfiles(
            features1.shapeProfile,
            features2.shapeProfile
        );

        const patternScore = this.compareRepetitionProfiles(
            features1.repetitionProfile,
            features2.repetitionProfile
        );

        const semanticScore = await this.compareDeepFeatures(
            features1.deepFeatures,
            features2.deepFeatures
        );

        // Calculate weighted final score
        const weights = {
            texture: 0.25,
            edge: 0.25,
            shape: 0.20,
            pattern: 0.15,
            semantic: 0.15
        };

        const score = (
            textureScore * weights.texture +
            edgeScore * weights.edge +
            shapeScore * weights.shape +
            patternScore * weights.pattern +
            semanticScore * weights.semantic
        );

        // Calculate confidence based on feature quality
        const confidence = this.calculateContentConfidence({
            textureScore,
            edgeScore,
            shapeScore,
            patternScore,
            semanticScore
        });

        return {
            score,
            confidence,
            details: {
                textureMatch: textureScore,
                edgeMatch: edgeScore,
                shapeMatch: shapeScore,
                patternMatch: patternScore,
                semanticMatch: semanticScore
            }
        };

    } catch (error) {
        logger.error('Error in content comparison:', error);
        return {
            score: 0,
            confidence: 0,
            details: {
                error: error.message,
                textureMatch: 0,
                edgeMatch: 0,
                shapeMatch: 0,
                patternMatch: 0,
                semanticMatch: 0
            }
        };
    }
}

// Helper methods for content comparison
static async compareTextureProfiles(profile1, profile2) {
    if (!profile1 || !profile2) return 0;

    const scores = {
        coarseness: 1 - Math.abs(profile1.coarseness - profile2.coarseness),
        contrast: 1 - Math.abs(profile1.contrast - profile2.contrast),
        directionality: 1 - Math.abs(profile1.directionality - profile2.directionality),
        linelikeness: 1 - Math.abs(profile1.linelikeness - profile2.linelikeness),
        regularity: 1 - Math.abs(profile1.regularity - profile2.regularity),
        roughness: 1 - Math.abs(profile1.roughness - profile2.roughness)
    };

    return Object.values(scores).reduce((sum, score) => sum + score, 0) / Object.keys(scores).length;
}

static async compareEdgeProfiles(profile1, profile2) {
    if (!profile1 || !profile2) return 0;

    const scores = {
        density: 1 - Math.abs(profile1.density - profile2.density),
        direction: 1 - Math.abs(profile1.direction - profile2.direction),
        sharpness: 1 - Math.abs(profile1.sharpness - profile2.sharpness),
        continuity: 1 - Math.abs(profile1.continuity - profile2.continuity)
    };

    return Object.values(scores).reduce((sum, score) => sum + score, 0) / Object.keys(scores).length;
}

static compareShapeProfiles(profile1, profile2) {
    if (!profile1 || !profile2) return 0;

    const geometricSimilarity = this.compareShapeSets(
        profile1.geometricShapes,
        profile2.geometricShapes
    );

    const organicSimilarity = this.compareShapeSets(
        profile1.organicShapes,
        profile2.organicShapes
    );

    const complexityDiff = Math.abs(profile1.complexity - profile2.complexity);

    return (
        geometricSimilarity * 0.4 +
        organicSimilarity * 0.4 +
        (1 - complexityDiff) * 0.2
    );
}

static compareRepetitionProfiles(profile1, profile2) {
    if (!profile1 || !profile2) return 0;

    const scores = {
        frequency: 1 - Math.abs(profile1.frequency - profile2.frequency),
        regularity: 1 - Math.abs(profile1.regularity - profile2.regularity),
        spacing: 1 - Math.abs(profile1.spacing - profile2.spacing)
    };

    return Object.values(scores).reduce((sum, score) => sum + score, 0) / Object.keys(scores).length;
}

static async compareDeepFeatures(features1, features2) {
    if (!features1 || !features2) return 0;

    const scores = {
        semantic: this.compareFeatureVectors(features1.semanticFeatures, features2.semanticFeatures),
        hierarchical: this.compareFeatureVectors(features1.hierarchicalFeatures, features2.hierarchicalFeatures),
        contextual: this.compareFeatureVectors(features1.contextualFeatures, features2.contextualFeatures)
    };

    return Object.values(scores).reduce((sum, score) => sum + score, 0) / Object.keys(scores).length;
}

static compareShapeSets(shapes1, shapes2) {
    if (!Array.isArray(shapes1) || !Array.isArray(shapes2)) return 0;
    if (shapes1.length === 0 || shapes2.length === 0) return 0;

    const matches = shapes1.map(shape1 => {
        return Math.max(...shapes2.map(shape2 => this.compareShapes(shape1, shape2)));
    });

    return matches.reduce((sum, score) => sum + score, 0) / matches.length;
}

static compareShapes(shape1, shape2) {
    if (!shape1 || !shape2) return 0;

    const areaRatio = Math.min(shape1.area, shape2.area) / Math.max(shape1.area, shape2.area);
    const perimeterRatio = Math.min(shape1.perimeter, shape2.perimeter) / Math.max(shape1.perimeter, shape2.perimeter);
    const circularityDiff = Math.abs(shape1.circularity - shape2.circularity);

    return (areaRatio * 0.4 + perimeterRatio * 0.3 + (1 - circularityDiff) * 0.3);
}

static compareFeatureVectors(vector1, vector2) {
    if (!vector1 || !vector2 || vector1.length !== vector2.length) return 0;

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vector1.length; i++) {
        dotProduct += vector1[i] * vector2[i];
        norm1 += vector1[i] * vector1[i];
        norm2 += vector2[i] * vector2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) return 0;
    return dotProduct / (norm1 * norm2);
}

static calculateContentConfidence(scores) {
    const validScores = Object.values(scores).filter(score => !isNaN(score) && score >= 0);
    if (validScores.length === 0) return 0;

    const averageScore = validScores.reduce((sum, score) => sum + score, 0) / validScores.length;
    const variance = validScores.reduce((sum, score) => sum + Math.pow(score - averageScore, 2), 0) / validScores.length;

    // Higher confidence if scores are consistently high with low variance
    const consistency = 1 - Math.sqrt(variance);
    return (averageScore * 0.7 + consistency * 0.3);
}

static async extractLocalFeatures(buffer) {
    // Implementation for feature extraction
    const image = sharp(buffer);
    const { data, info } = await image
        .greyscale()
        .raw()
        .toBuffer({ resolveWithObject: true });
    
    return this.detectKeypoints(data, info.width, info.height);
}

static async detectKeypoints(data, width, height) {
    const keypoints = [];
    const blockSize = 16;
    
    for (let y = 0; y < height; y += blockSize) {
        for (let x = 0; x < width; x += blockSize) {
            let blockSum = 0;
            for (let by = 0; by < blockSize && (y + by) < height; by++) {
                for (let bx = 0; bx < blockSize && (x + bx) < width; bx++) {
                    const idx = ((y + by) * width + (x + bx));
                    blockSum += data[idx];
                }
            }
            keypoints.push({
                x: x + blockSize/2,
                y: y + blockSize/2,
                value: blockSum / (blockSize * blockSize)
            });
        }
    }
    return keypoints;
}

static calculateFeatureDistance(f1, f2) {
    return Math.abs(f1.value - f2.value);
}

static compareMetadata(metadata1, metadata2) {
    if (!metadata1 || !metadata2) return 0;
    
    let score = 0;
    if (metadata1.format === metadata2.format) score += 0.5;
    if (metadata1.space === metadata2.space) score += 0.5;
    
    return score;
}

static findHistogramPeaks(histogram, minProminence = 0.1) {
    const peaks = [];
    const smoothed = this.smoothHistogram(histogram);
    
    for (let i = 1; i < smoothed.length - 1; i++) {
        if (smoothed[i] > smoothed[i-1] && smoothed[i] > smoothed[i+1]) {
            const prominence = Math.min(
                smoothed[i] - smoothed[i-1],
                smoothed[i] - smoothed[i+1]
            );
            if (prominence >= minProminence) {
                peaks.push({ position: i, value: smoothed[i], prominence });
            }
        }
    }
    return peaks;
}

static smoothHistogram(histogram, windowSize = 3) {
    const smoothed = [...histogram];
    for (let i = windowSize; i < histogram.length - windowSize; i++) {
        let sum = 0;
        for (let j = -windowSize; j <= windowSize; j++) {
            sum += histogram[i + j];
        }
        smoothed[i] = sum / (2 * windowSize + 1);
    }
    return smoothed;
}

static compareColorPeaks(peaks1, peaks2) {
    if (!peaks1.length || !peaks2.length) return 0;
    
    const matches = peaks1.map(p1 => {
        const bestMatch = peaks2.reduce((best, p2) => {
            const distance = Math.abs(p1.position - p2.position);
            const valueDiff = Math.abs(p1.value - p2.value);
            const score = 1 - (distance / 255) * 0.7 - (valueDiff / 255) * 0.3;
            return score > best.score ? { score, peak: p2 } : best;
        }, { score: 0, peak: null });
        return bestMatch.score;
    });
    
    return matches.reduce((sum, score) => sum + score, 0) / matches.length;
}

static async matchFeatures(features1, features2) {
    // Implementation for feature matching
    return features1.map(f1 => {
        const matches = features2
            .map(f2 => ({
                distance: this.calculateFeatureDistance(f1, f2),
                feature: f2
            }))
            .sort((a, b) => a.distance - b.distance);
        
        return matches[0];
    }).filter(m => m.distance < 0.8);
}

    static async performDetailedComparison(sig1, sig2, options) {
        const results = {
            perceptualHash: await this.comparePerceptualHashes(sig1.perceptualHashes, sig2.perceptualHashes),
            colorMetrics: this.compareColorMetrics(sig1.colorMetrics, sig2.colorMetrics),
            edgeSignature: this.compareEdgeSignatures(sig1.edgeSignature, sig2.edgeSignature),
            structural: this.compareStructuralFeatures(sig1.metadata, sig2.metadata),
            metadata: this.compareMetadata(sig1.metadata, sig2.metadata)
        };

        // Enhanced analysis for high-scoring matches
        if (this.isHighScoringMatch(results)) {
            results.enhancedAnalysis = await this.performEnhancedAnalysis(sig1, sig2, options);
        }

        return results;
    }

    static async comparePerceptualHashes(hashes1, hashes2) {
        if (!hashes1?.length || !hashes2?.length) return 0;

        const similarities = await Promise.all(
            hashes1.map(async (hash1, i) => {
                const hash2 = hashes2[i];
                if (!hash2) return 0;
                return this.calculateHashSimilarity(hash1.hash, hash2.hash);
            })
        );

        return similarities.reduce((sum, score) => sum + score, 0) / similarities.length;
    }

static compareColorMetrics(metrics1, metrics2) {
    if (!metrics1 || !metrics2) return 0;
    
    const scores = {
        means: this.compareColorChannels(metrics1.means, metrics2.means),
        histogram: this.compareHistograms(metrics1.histogram, metrics2.histogram),
        distribution: this.compareColorDistribution(metrics1, metrics2),
        dominantColors: this.compareDominantColors(metrics1, metrics2),
        colorVariance: this.compareColorVariance(metrics1, metrics2)
    };

    const weights = {
        means: 0.3,
        histogram: 0.25,
        distribution: 0.2,
        dominantColors: 0.15,
        colorVariance: 0.1
    };

    return Object.entries(scores)
        .reduce((sum, [key, score]) => sum + score * weights[key], 0);
}

// Add these helper methods right after compareColorMetrics
static compareDominantColors(metrics1, metrics2) {
    // Implementation for dominant color comparison
    if (!metrics1?.histogram?.data || !metrics2?.histogram?.data) return 0;
    
    const peaks1 = this.findHistogramPeaks(metrics1.histogram.data);
    const peaks2 = this.findHistogramPeaks(metrics2.histogram.data);
    
    return this.compareColorPeaks(peaks1, peaks2);
}

static compareColorVariance(metrics1, metrics2) {
    if (!metrics1?.stdDevs || !metrics2?.stdDevs) return 0;
    
    return 1 - Math.abs(
        metrics1.stdDevs.reduce((a, b) => a + b, 0) -
        metrics2.stdDevs.reduce((a, b) => a + b, 0)
    ) / (255 * 3);
}

    static compareColorChannels(channels1, channels2) {
        if (!channels1?.length || !channels2?.length) return 0;
        
        const diffs = channels1.map((val, i) => {
            const diff = Math.abs(val - (channels2[i] || 0)) / 255;
            return 1 - diff;
        });

        return diffs.reduce((sum, score) => sum + score, 0) / diffs.length;
    }

    static compareHistograms(hist1, hist2) {
        if (!hist1?.data || !hist2?.data || hist1.bins !== hist2.bins) return 0;

        const intersection = hist1.data.reduce((sum, count, i) => {
            const minCount = Math.min(
                count / hist1.totalPixels,
                hist2.data[i] / hist2.totalPixels
            );
            return sum + minCount;
        }, 0);

        return intersection;
    }

    static compareEdgeSignatures(sig1, sig2) {
        if (!sig1?.length || !sig2?.length) return 0;

        const minLength = Math.min(sig1.length, sig2.length);
        let similarity = 0;

        for (let i = 0; i < minLength; i++) {
            similarity += 1 - Math.abs(sig1[i] - sig2[i]);
        }

        return similarity / minLength;
    }

    static compareStructuralFeatures(metadata1, metadata2) {
        if (!metadata1?.dimensions || !metadata2?.dimensions) return 0;

        const aspectRatioDiff = Math.abs(
            metadata1.dimensions.aspectRatio - metadata2.dimensions.aspectRatio
        );

        if (aspectRatioDiff > COMPARISON_THRESHOLDS.ASPECT_RATIO_TOLERANCE) {
            return 0;
        }

        return 1 - (aspectRatioDiff / COMPARISON_THRESHOLDS.ASPECT_RATIO_TOLERANCE);
    }

static async validateMatch(searchImage, candidateImage, comparisonResults, strictMode = true) {
    const scales = [0.5, 1.0, 2.0];
    const validationResults = await Promise.all(scales.map(async scale => {
        const scaledSearch = await this.scaleImage(searchImage, scale);
        const scaledCandidate = await this.scaleImage(candidateImage, scale);
        
        const basicChecks = {
            dimensionsValid: this.validateDimensions(comparisonResults.structural),
            colorProfileValid: this.validateColorProfile(comparisonResults.colorMetrics),
            edgePatternsValid: this.validateEdgePatterns(comparisonResults.edgeSignature),
            metadataValid: this.validateMetadata(comparisonResults.metadata)
        };

        return {
            scale,
            checks: basicChecks,
            score: Object.values(basicChecks).filter(Boolean).length / Object.keys(basicChecks).length
        };
    }));

    const combinedScore = validationResults.reduce((sum, result) => 
        sum + result.score, 0) / validationResults.length;

    return {
        isValid: combinedScore >= COMPARISON_THRESHOLDS.MIN_VALIDATION_SCORE,
        score: combinedScore,
        checks: validationResults,
        confidence: this.calculateValidationConfidence(validationResults)
    };
}

    static calculateFinalScore(results, confidenceLevels) {
        const weightedScores = Object.entries(results).map(([metric, score]) => ({
            score: score * (COMPARISON_WEIGHTS[metric.toUpperCase()] || 0),
            confidence: confidenceLevels[metric] || 0
        }));

        const totalWeight = weightedScores.reduce((sum, { confidence }) => sum + confidence, 0);
        
        return weightedScores.reduce((sum, { score, confidence }) => 
            sum + (score * confidence), 0) / totalWeight;
    }

    static determineMatchType(finalScore, validationResult) {
        if (finalScore > COMPARISON_THRESHOLDS.EXACT_MATCH && validationResult.isValid) {
            return 'exact';
        }
        if (finalScore > COMPARISON_THRESHOLDS.HIGH_CONFIDENCE) {
            return 'very_high';
        }
        if (finalScore > COMPARISON_THRESHOLDS.MEDIUM_CONFIDENCE) {
            return 'high';
        }
        if (finalScore > COMPARISON_THRESHOLDS.LOW_CONFIDENCE) {
            return 'medium';
        }
        return 'low';
    }

    static async findSequences(images, baseImage = null) {
        const sequences = [];
        const processed = new Set();

        for (const image of images) {
            if (processed.has(image.path)) continue;

            const sequence = await this.buildSequence(image, images, baseImage);
            
            if (sequence.length > 1) {
                sequences.push({
                    images: sequence,
                    metrics: await this.calculateSequenceMetrics(sequence),
                    confidence: await this.calculateSequenceConfidence(sequence)
                });

                sequence.forEach(img => processed.add(img.path));
            }
        }

        return sequences;
    }

    static async buildSequence(startImage, allImages, baseImage = null) {
        const sequence = [startImage];
        const potentialMatches = allImages.filter(img => img.path !== startImage.path);

        for (const candidate of potentialMatches) {
            const similarity = await this.compareImages(startImage.data, candidate.data);
            
            if (similarity.similarity >= COMPARISON_THRESHOLDS.SEQUENCE_SIMILARITY) {
                sequence.push({
                    ...candidate,
                    similarity: similarity.similarity,
                    matchConfidence: similarity.confidence
                });
            }
        }

        // Sort by timestamp if available
        sequence.sort((a, b) => {
            const timeA = new Date(a.metadata?.timestamp || 0);
            const timeB = new Date(b.metadata?.timestamp || 0);
            return timeA - timeB;
        });

        return sequence;
    }

    static async calculateSequenceMetrics(sequence) {
        const metrics = {
            totalChanges: 0,
            averageChange: 0,
            timespan: 0,
            progressionRate: 0
        };

        if (sequence.length < 2) return metrics;

        // Calculate changes between successive images
        for (let i = 1; i < sequence.length; i++) {
            const comparison = await this.compareImages(
                sequence[i - 1].data,
                sequence[i].data,
                { includeDetails: true }
            );

            metrics.totalChanges += 1 - comparison.similarity;
        }

        metrics.averageChange = metrics.totalChanges / (sequence.length - 1);
        metrics.timespan = new Date(sequence[sequence.length - 1].metadata.timestamp) -
                          new Date(sequence[0].metadata.timestamp);
        metrics.progressionRate = metrics.totalChanges / (metrics.timespan / (1000 * 60 * 60 * 24));

        return metrics;
    }

    static async performEnhancedValidation(searchImage, candidateImage, baseResults) {
        const enhancedChecks = {
            featureMatching: await this.performFeatureMatching(searchImage, candidateImage),
            patternAnalysis: await this.analyzePatternSimilarity(searchImage, candidateImage),
            contextualAnalysis: await this.analyzeImageContext(searchImage, candidateImage)
        };

        return {
            checks: enhancedChecks,
            score: Object.values(enhancedChecks)
                .filter(Boolean)
                .length / Object.keys(enhancedChecks).length
        };
    }

    static calculateValidationConfidence(basicChecks, enhancedValidation) {
        const basicScore = Object.values(basicChecks)
            .filter(Boolean)
            .length / Object.keys(basicChecks).length;

        if (!enhancedValidation) return basicScore;

        return (basicScore + enhancedValidation.score) / 2;
    }

static isHighScoringMatch(results) {
        return Object.values(results).some(score => score > 0.9);
    }

    static async performEnhancedAnalysis(sig1, sig2, options) {
        return {
            featureMatching: await this.performFeatureMatching(sig1, sig2),
            patternAnalysis: await this.analyzePatternSimilarity(sig1, sig2),
            contextualAnalysis: await this.analyzeImageContext(sig1, sig2)
        };
    }

    static async analyzePatternSimilarity(img1, img2) {
        const [patterns1, patterns2] = await Promise.all([
            this.extractPatterns(img1),
            this.extractPatterns(img2)
        ]);
        return this.comparePatterns(patterns1, patterns2);
    }

    static async analyzeImageContext(img1, img2) {
        return {
            spatialRelations: await this.compareSpatialRelations(img1, img2),
            semanticContext: await this.compareSemanticContext(img1, img2)
        };
    }

    static calculateConfidenceLevels(results) {
        return Object.entries(results).reduce((levels, [metric, value]) => {
            levels[metric] = this.calculateMetricConfidence(value);
            return levels;
        }, {});
    }

    static calculateMetricConfidence(value) {
        if (typeof value === 'number') {
            return value > 0.95 ? 1.0 :
                   value > 0.85 ? 0.8 :
                   value > 0.75 ? 0.6 :
                   0.4;
        }
        return 0.5;
    }

    static async scaleImage(buffer, scale) {
        return await sharp(buffer)
            .resize({
                width: Math.round(scale * 100),
                height: Math.round(scale * 100),
                fit: 'contain'
            })
            .toBuffer();
    }

    static validateDimensions(structuralResults) {
        return structuralResults > COMPARISON_THRESHOLDS.ASPECT_RATIO_TOLERANCE;
    }

    static validateColorProfile(colorResults) {
        return colorResults > COMPARISON_THRESHOLDS.COLOR_TOLERANCE;
    }

    static validateEdgePatterns(edgeResults) {
        return edgeResults > COMPARISON_THRESHOLDS.LOW_CONFIDENCE;
    }

    static validateMetadata(metadataResults) {
        return metadataResults > COMPARISON_THRESHOLDS.MIN_VALIDATION_SCORE;
    }

    static async extractPatterns(buffer) {
        const image = sharp(buffer);
        const { data, info } = await image
            .greyscale()
            .raw()
            .toBuffer({ resolveWithObject: true });
        return this.detectPatterns(data, info.width, info.height);
    }

    static async detectPatterns(data, width, height) {
        // Implement pattern detection based on your specific needs
        return {
            edges: this.detectEdgePatterns(data, width, height),
            texture: this.analyzeTexture(data, width, height),
            gradients: this.analyzeGradients(data, width, height)
        };
    }

    static detectEdgePatterns(data, width, height) {
        // Simplified edge detection
        let edges = 0;
        for (let i = 1; i < height - 1; i++) {
            for (let j = 1; j < width - 1; j++) {
                const idx = i * width + j;
                const diff = Math.abs(data[idx] - data[idx + 1]) +
                           Math.abs(data[idx] - data[idx + width]);
                if (diff > 30) edges++;
            }
        }
        return edges / (width * height);
    }

    static analyzeTexture(data, width, height) {
        // Simplified texture analysis
        let variance = 0;
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        for (const pixel of data) {
            variance += (pixel - mean) ** 2;
        }
        return Math.sqrt(variance / data.length) / 255;
    }

    static analyzeGradients(data, width, height) {
        // Simplified gradient analysis
        let gradients = 0;
        for (let i = 0; i < height - 1; i++) {
            for (let j = 0; j < width - 1; j++) {
                const idx = i * width + j;
                const gradient = Math.sqrt(
                    (data[idx + 1] - data[idx]) ** 2 +
                    (data[idx + width] - data[idx]) ** 2
                );
                gradients += gradient;
            }
        }
        return gradients / (width * height * 255);
    }

// Add at the end of the ImageComparison class, before the closing }
    static async analyzePatterns(buffer) {
        const image = sharp(buffer);
        
        const [
            textures,
            edges,
            shapes,
            repetitions
        ] = await Promise.all([
            this.analyzeTextures(image),
            this.detectEdges(image),
            this.detectShapes(image),
            this.findRepetitivePatterns(image)
        ]);

        const deepFeatures = await this.extractDeepFeatures(buffer);

        return {
            textureProfile: {
                coarseness: textures.coarseness,
                contrast: textures.contrast,
                directionality: textures.directionality,
                linelikeness: textures.linelikeness,
                regularity: textures.regularity,
                roughness: textures.roughness
            },
            edgeProfile: {
                density: edges.density,
                direction: edges.direction,
                sharpness: edges.sharpness,
                continuity: edges.continuity
            },
            shapeProfile: {
                geometricShapes: shapes.geometric,
                organicShapes: shapes.organic,
                complexity: shapes.complexity
            },
            repetitionProfile: {
                frequency: repetitions.frequency,
                regularity: repetitions.regularity,
                spacing: repetitions.spacing
            },
            deepFeatures: {
                semanticFeatures: deepFeatures.semantic,
                hierarchicalFeatures: deepFeatures.hierarchical,
                contextualFeatures: deepFeatures.contextual
            }
        };
    }

    static calculateConfidenceThreshold(quality) {
        return Math.min(
            0.95,
            0.8 + (quality.sharpness * 0.1) +
                 (quality.contrast * 0.1) +
                 (quality.snr * 0.1)
        );
    }

static calculateContrast(stats) {
        return stats.channels.reduce((sum, channel) => 
            sum + (channel.max - channel.min) / 255, 0) / stats.channels.length;
    }

    static estimateSharpness(stats) {
        return stats.channels.reduce((sum, channel) => 
            sum + channel.stdev / 255, 0) / stats.channels.length;
    }

    static calculateSNR(stats) {
        return stats.channels.reduce((sum, channel) => 
            sum + (channel.mean / (channel.stdev || 1)), 0) / stats.channels.length;
    }

}

module.exports = ImageComparison;
