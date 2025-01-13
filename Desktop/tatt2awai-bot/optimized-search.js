const logger = require('./logger');
const EnhancedSignatureStore = require('./enhanced-store');
const EnhancedSignatureGenerator = require('./enhanced-signature-generator');
const dropboxManager = require('./dropboxManager');

class OptimizedImageSearch {
    constructor() {
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;

        logger.info('Initializing visual search system...', {
            service: "tatt2awai-bot"
        });

        try {
            await EnhancedSignatureStore.initialize();
            this.initialized = true;
            logger.info('Visual search system initialized successfully', {
                service: "tatt2awai-bot"
            });
        } catch (error) {
            logger.error('Failed to initialize visual search:', {
                service: "tatt2awai-bot",
                error: error.message,
                stack: error.stack
            });
            throw error;
        }
    }

    async search(searchBuffer) {
        try {
            // Ensure system is initialized
            await this.initialize();

            const startTime = Date.now();

            // Generate signature for search image
            const searchSignature = await EnhancedSignatureGenerator.generateSignature(searchBuffer);

            // Get all images from Dropbox
            const allImages = await dropboxManager.fetchDropboxEntries('');
            if (!allImages?.result?.entries) {
                throw new Error('Failed to fetch images from Dropbox');
            }

            // Multi-stage matching process
            // Stage 1: Quick filtering using mini hashes
            const quickCandidates = allImages.result.entries.filter(image => {
                const signature = EnhancedSignatureStore.signatures.get(image.path_lower);
                if (!signature) return false;

                const miniHashScore = this.compareHashes(
                    searchSignature.perceptualHashes.mini,
                    signature.perceptualHashes.mini
                );

                return miniHashScore > 0.8;
            });

            logger.info('Quick filtering results:', {
                service: "tatt2awai-bot",
                candidates: quickCandidates.length,
                timeElapsed: Date.now() - startTime
            });

            if (quickCandidates.length === 0) {
                return {
                    success: false,
                    message: 'No matching image found',
                    timeTaken: Date.now() - startTime
                };
            }

            // Stage 2: Detailed comparison
            const detailedResults = await Promise.all(
                quickCandidates.map(async candidate => {
                    const signature = EnhancedSignatureStore.signatures.get(candidate.path_lower);
                    const similarity = await this.compareSignaturesDetailed(searchSignature, signature);

                    return {
                        path: candidate.path_lower,
                        ...similarity,
                        metadata: candidate
                    };
                })
            );

            // Find best match
            const matches = detailedResults
                .filter(result => result.score > 0.85)
                .sort((a, b) => b.score - a.score);

            if (matches.length === 0) {
                return {
                    success: false,
                    message: 'No matching image found',
                    timeTaken: Date.now() - startTime
                };
            }

            // Get best match
            const bestMatch = matches[0];

            // For exact matches, find sequence
            let sequence = null;
            if (bestMatch.score > 0.98) {
                sequence = await this.findImageSequence(bestMatch, allImages.result.entries);
            }

            const timeTaken = Date.now() - startTime;
            logger.info('Search completed', {
                service: "tatt2awai-bot",
                matchFound: true,
                score: bestMatch.score,
                timeTaken
            });

            return {
                success: true,
                match: {
                    path: bestMatch.path,
                    confidence: bestMatch.confidence,
                    similarity: bestMatch.score,
                    matchType: bestMatch.score > 0.98 ? 'exact' : 'similar',
                    details: bestMatch.components
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
                timeTaken
            };

        } catch (error) {
            logger.error('Search failed:', {
                service: "tatt2awai-bot",
                error: error.message,
                stack: error.stack
            });
            throw error;
        }
    }

    async compareSignaturesDetailed(sig1, sig2) {
        try {
            const scores = {
                // Perceptual hash comparison at multiple scales
                miniHash: this.compareHashes(sig1.perceptualHashes.mini, sig2.perceptualHashes.mini),
                normalHash: this.compareHashes(sig1.perceptualHashes.normal, sig2.perceptualHashes.normal),
                macroHash: this.compareHashes(sig1.perceptualHashes.macro, sig2.perceptualHashes.macro),
                dctHash: this.compareDCTHashes(sig1.perceptualHashes.normal.dct, sig2.perceptualHashes.normal.dct),
                
                // Edge and color signature comparison
                edges: this.compareEdgeSignatures(sig1.edgeSignature, sig2.edgeSignature),
                colors: this.compareColorSignatures(sig1.colorSignature, sig2.colorSignature),
                
                // Local feature comparison
                features: this.compareLocalFeatures(sig1.features, sig2.features),
                
                // Metadata comparison
                metadata: this.compareMetadata(sig1.metadata, sig2.metadata)
            };

            // Weight the scores based on their reliability
            const weights = {
                miniHash: 0.1,
                normalHash: 0.2,
                macroHash: 0.1,
                dctHash: 0.1,
                edges: 0.15,
                colors: 0.15,
                features: 0.15,
                metadata: 0.05
            };

            const weightedScore = Object.entries(scores).reduce((total, [key, score]) => 
                total + score * weights[key], 0);

            const confidence = this.calculateConfidence(scores);

            // Geometric consistency check for high-scoring matches
            if (weightedScore > 0.9) {
                const geometricScore = await this.verifyGeometricConsistency(sig1, sig2);
                if (geometricScore < 0.5) {
                    return {
                        score: weightedScore * 0.5,
                        confidence: confidence * 0.5,
                        components: scores
                    };
                }
            }

            return {
                score: weightedScore,
                confidence,
                components: scores
            };

        } catch (error) {
            logger.error('Error comparing signatures:', {
                service: "tatt2awai-bot",
                error: error.message
            });
            return {
                score: 0,
                confidence: 0,
                components: {}
            };
        }
    }

    compareHashes(hash1, hash2) {
        if (!hash1?.binary || !hash2?.binary) return 0;
        
        let diff = 0;
        for (let i = 0; i < hash1.binary.length; i++) {
            if (hash1.binary[i] !== hash2.binary[i]) diff++;
        }
        return 1 - (diff / hash1.binary.length);
    }

    compareDCTHashes(dct1, dct2) {
        if (!dct1 || !dct2 || !Array.isArray(dct1.dctCoefficients) || !Array.isArray(dct2.dctCoefficients)) {
            return 0;
        }

        const minLength = Math.min(dct1.dctCoefficients.length, dct2.dctCoefficients.length);
        let diff = 0;
        
        for (let i = 0; i < minLength; i++) {
            diff += Math.abs(dct1.dctCoefficients[i] - dct2.dctCoefficients[i]);
        }

        return Math.max(0, 1 - (diff / minLength / 255));
    }

    compareEdgeSignatures(edge1, edge2) {
        if (!edge1?.multi_scale || !edge2?.multi_scale) return 0;

        const orientationMatch = Math.abs(
            edge1.dominant_orientation.angle - edge2.dominant_orientation.angle
        ) <= 45;

        const histogramSimilarity = edge1.multi_scale.reduce((sum, scale, idx) => {
            const correspondingScale = edge2.multi_scale[idx];
            if (!correspondingScale) return sum;

            const histSimilarity = this.compareHistograms(
                scale.signature,
                correspondingScale.signature
            );
            const orientSimilarity = Math.abs(
                scale.orientation.dominant - correspondingScale.orientation.dominant
            ) / 180;

            return sum + (histSimilarity * 0.7 + (1 - orientSimilarity) * 0.3);
        }, 0) / edge1.multi_scale.length;

        return orientationMatch ? histogramSimilarity : histogramSimilarity * 0.5;
    }

    compareColorSignatures(color1, color2) {
        if (!color1 || !color2) return 0;

        const histSimilarity = this.compareHistograms(color1.histogram, color2.histogram);
        const dominantSimilarity = this.compareDominantColors(color1.dominantColors, color2.dominantColors);
        const avgColorSimilarity = this.compareAverageColors(color1.averageColor, color2.averageColor);

        return (histSimilarity * 0.4 + dominantSimilarity * 0.4 + avgColorSimilarity * 0.2);
    }

    compareHistograms(hist1, hist2) {
        if (!Array.isArray(hist1) || !Array.isArray(hist2)) return 0;
        
        const minLength = Math.min(hist1.length, hist2.length);
        let similarity = 0;
        
        for (let i = 0; i < minLength; i++) {
            similarity += Math.min(hist1[i], hist2[i]);
        }
        
        return similarity;
    }

    compareDominantColors(colors1, colors2) {
        if (!Array.isArray(colors1) || !Array.isArray(colors2)) return 0;

        let totalSimilarity = 0;
        let maxSimilarity = Math.max(colors1.length, colors2.length);

        colors1.forEach(color1 => {
            const similarities = colors2.map(color2 => 
                this.compareRGBColors(color1.rgb, color2.rgb) * 
                Math.min(color1.prominence, color2.prominence)
            );
            totalSimilarity += Math.max(...similarities);
        });

        return totalSimilarity / maxSimilarity;
    }

    compareRGBColors(rgb1, rgb2) {
        const diff = Math.sqrt(
            Math.pow(rgb1.r - rgb2.r, 2) +
            Math.pow(rgb1.g - rgb2.g, 2) +
            Math.pow(rgb1.b - rgb2.b, 2)
        );
        return 1 - (diff / (Math.sqrt(3) * 255));
    }

    compareAverageColors(avg1, avg2) {
        if (!avg1 || !avg2) return 0;
        return this.compareRGBColors(avg1, avg2);
    }

    compareLocalFeatures(features1, features2) {
        if (!Array.isArray(features1) || !Array.isArray(features2)) return 1.0;

        const matches = this.matchLocalFeatures(features1, features2);
        if (matches.length < 4) return 1.0;

        const consistentMatches = this.ransacTransform(matches);
        return consistentMatches.length / matches.length;
    }

    matchLocalFeatures(features1, features2) {
        const matches = [];
        const threshold = 0.8;

        for (const f1 of features1) {
            let bestMatch = { distance: Infinity, feature: null };
            let secondBest = { distance: Infinity, feature: null };

            for (const f2 of features2) {
                const distance = this.calculateFeatureDistance(f1.descriptor, f2.descriptor);
                if (distance < bestMatch.distance) {
                    secondBest = bestMatch;
                    bestMatch = { distance, feature: f2 };
                } else if (distance < secondBest.distance) {
                    secondBest = { distance, feature: f2 };
                }
            }

            if (bestMatch.distance < threshold * secondBest.distance) {
                matches.push({
                    from: f1,
                    to: bestMatch.feature,
                    distance: bestMatch.distance
                });
            }
        }

        return matches;
    }

    calculateFeatureDistance(desc1, desc2) {
        if (!desc1 || !desc2) return Infinity;
        
        let sum = 0;
        const length = Math.min(desc1.length, desc2.length);
        
        for (let i = 0; i < length; i++) {
            sum += Math.pow(desc1[i] - desc2[i], 2);
        }
        
        return Math.sqrt(sum);
    }

    compareMetadata(meta1, meta2) {
        if (!meta1?.dimensions || !meta2?.dimensions) return 0;
        
        const aspectRatioDiff = Math.abs(
            meta1.dimensions.aspectRatio - meta2.dimensions.aspectRatio
        );
        
        if (aspectRatioDiff > 0.1) return 0;
        
        let qualityScore = 1;
        if (meta1.characteristics?.quality && meta2.characteristics?.quality) {
            qualityScore = 1 - Math.abs(
                meta1.characteristics.quality.overall - 
                meta2.characteristics.quality.overall
            );
        }
        
        return (1 - (aspectRatioDiff * 10)) * qualityScore;
    }

    calculateConfidence(scores) {
        const values = Object.values(scores).filter(score => !isNaN(score));
        if (values.length === 0) return 0;

        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => 
            sum + Math.pow(val - mean, 2), 0) / values.length;

        // Higher confidence if scores are consistently high
        const consistency = 1 - Math.sqrt(variance);
        const baseConfidence = mean * consistency;

        // Boost confidence if all scores are high
        const allScoresHigh = values.every(val => val > 0.9);
        
return allScoresHigh ? 
            Math.min(baseConfidence * 1.2, 1) : baseConfidence;
    }

    async verifyGeometricConsistency(sig1, sig2) {
        if (!sig1.features || !sig2.features) return 1.0;

        const matches = this.matchLocalFeatures(sig1.features, sig2.features);
        if (matches.length < 4) return 1.0;

        const consistentMatches = this.ransacTransform(matches);
        return consistentMatches.length / matches.length;
    }

    ransacTransform(matches, iterations = 1000, threshold = 3.0) {
        if (matches.length < 4) return matches;

        let bestInliers = [];
        let bestCount = 0;

        for (let i = 0; i < iterations; i++) {
            // Select random matches
            const sample = this.getRandomSample(matches, 4);
            const transform = this.estimateTransform(sample);
            if (!transform) continue;

            // Count inliers
            const inliers = matches.filter(match => {
                const error = this.calculateTransformError(match, transform);
                return error < threshold;
            });

            if (inliers.length > bestCount) {
                bestCount = inliers.length;
                bestInliers = inliers;
            }
        }

        return bestInliers;
    }

    getRandomSample(matches, count) {
        const sample = [];
        const used = new Set();

        while (sample.length < count && sample.length < matches.length) {
            const idx = Math.floor(Math.random() * matches.length);
            if (!used.has(idx)) {
                used.add(idx);
                sample.push(matches[idx]);
            }
        }

        return sample;
    }

    estimateTransform(matches) {
        if (matches.length < 4) return null;

        // Calculate centroid
        const srcPoints = matches.map(m => ({ x: m.from.x, y: m.from.y }));
        const dstPoints = matches.map(m => ({ x: m.to.x, y: m.to.y }));

        const srcCentroid = this.calculateCentroid(srcPoints);
        const dstCentroid = this.calculateCentroid(dstPoints);

        // Normalize points
        const srcNorm = this.normalizePoints(srcPoints, srcCentroid);
        const dstNorm = this.normalizePoints(dstPoints, dstCentroid);

        // Calculate transformation matrix
        return {
            translation: {
                x: dstCentroid.x - srcCentroid.x,
                y: dstCentroid.y - srcCentroid.y
            },
            scale: this.calculateScale(srcNorm, dstNorm),
            rotation: this.calculateRotation(srcNorm, dstNorm)
        };
    }

    calculateCentroid(points) {
        const sum = points.reduce((acc, p) => ({
            x: acc.x + p.x,
            y: acc.y + p.y
        }), { x: 0, y: 0 });

        return {
            x: sum.x / points.length,
            y: sum.y / points.length
        };
    }

    normalizePoints(points, centroid) {
        return points.map(p => ({
            x: p.x - centroid.x,
            y: p.y - centroid.y
        }));
    }

    calculateScale(srcPoints, dstPoints) {
        const srcDist = Math.sqrt(srcPoints.reduce((sum, p) => 
            sum + (p.x * p.x + p.y * p.y), 0) / srcPoints.length);
        const dstDist = Math.sqrt(dstPoints.reduce((sum, p) => 
            sum + (p.x * p.x + p.y * p.y), 0) / dstPoints.length);
        
        return dstDist / srcDist;
    }

    calculateRotation(srcPoints, dstPoints) {
        let sumCos = 0;
        let sumSin = 0;

        for (let i = 0; i < srcPoints.length; i++) {
            const srcMag = Math.sqrt(srcPoints[i].x * srcPoints[i].x + srcPoints[i].y * srcPoints[i].y);
            const dstMag = Math.sqrt(dstPoints[i].x * dstPoints[i].x + dstPoints[i].y * dstPoints[i].y);
            
            if (srcMag > 0 && dstMag > 0) {
                sumCos += (srcPoints[i].x * dstPoints[i].x + srcPoints[i].y * dstPoints[i].y) / (srcMag * dstMag);
                sumSin += (srcPoints[i].x * dstPoints[i].y - srcPoints[i].y * dstPoints[i].x) / (srcMag * dstMag);
            }
        }

        return Math.atan2(sumSin, sumCos);
    }

    calculateTransformError(match, transform) {
        // Apply transform to source point
        const srcNorm = {
            x: match.from.x - transform.translation.x,
            y: match.from.y - transform.translation.y
        };

        const cos = Math.cos(transform.rotation);
        const sin = Math.sin(transform.rotation);

        const transformed = {
            x: transform.scale * (srcNorm.x * cos - srcNorm.y * sin),
            y: transform.scale * (srcNorm.x * sin + srcNorm.y * cos)
        };

        // Calculate error
        return Math.sqrt(
            Math.pow(transformed.x - match.to.x, 2) +
            Math.pow(transformed.y - match.to.y, 2)
        );
    }

    async findImageSequence(matchedImage, allImages) {
        const directory = path.dirname(matchedImage.path);
        const dirImages = allImages.filter(img => path.dirname(img.path_lower) === directory)
            .sort((a, b) => {
                // First try sequence numbers in filename
                const seqA = this.extractSequenceNumber(a.name);
                const seqB = this.extractSequenceNumber(b.name);
                if (seqA !== null && seqB !== null) return seqA - seqB;

                // Fall back to timestamps
                return new Date(a.server_modified) - new Date(b.server_modified);
            });

        const matchIndex = dirImages.findIndex(img => img.path_lower === matchedImage.path);
        if (matchIndex === -1) return null;

        return {
            current: matchIndex,
            total: dirImages.length,
            before: dirImages.slice(0, matchIndex),
            after: dirImages.slice(matchIndex + 1)
        };
    }

    extractSequenceNumber(filename) {
        const patterns = [
            /t(\d+)/i,                    // t1, t2, etc.
            /session[_-]?(\d+)/i,         // session1, session_2
            /treatment[_-]?(\d+)/i,       // treatment1, treatment_2
            /before|after/i,              // Special handling for before/after
        ];

        for (const pattern of patterns) {
            const match = filename.match(pattern);
            if (match) {
                if (match[1]) return parseInt(match[1]);
                if (match[0].toLowerCase() === 'before') return 0;
                if (match[0].toLowerCase() === 'after') return 9999;
            }
        }
        return null;
    }

    getStatus() {
        return {
            initialized: this.initialized,
            storeStatus: EnhancedSignatureStore.getStatus()
        };
    }
}

module.exports = new OptimizedImageSearch();
