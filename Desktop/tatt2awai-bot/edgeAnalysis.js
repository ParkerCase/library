// edgeAnalysis.js - Enhanced Production Version
const sharp = require('sharp');
const _ = require('lodash');
const mathUtils = require('./mathUtils');
const logger = require('./logger');
const monitor = require('./monitoring');
const { ValidationError, ProcessingError } = require('./errorHandler');



class EdgeAnalysis {
    // Enhanced kernels for edge detection
    static GAUSSIAN_KERNELS = {
        3: [
            [0.075, 0.124, 0.075],
            [0.124, 0.204, 0.124],
            [0.075, 0.124, 0.075]
        ],
        5: [
            [0.003, 0.013, 0.022, 0.013, 0.003],
            [0.013, 0.059, 0.097, 0.059, 0.013],
            [0.022, 0.097, 0.159, 0.097, 0.022],
            [0.013, 0.059, 0.097, 0.059, 0.013],
            [0.003, 0.013, 0.022, 0.013, 0.003]
        ]
    };

    static SOBEL_KERNELS = {
        horizontal: [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        vertical: [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]
    };

    // Configuration constants
    static CANNY_THRESHOLDS = {
        LOW: 0.1,
        HIGH: 0.3
    };

    static ORIENTATION_BINS = 36; // 5-degree precision
    static MIN_EDGE_LENGTH = 5;
    static MAX_GAP_LENGTH = 2;

static async analyzeEdges(buffer) {
    const opId = monitor.startOperation('edgeAnalysis');
    
    try {
        if (!Buffer.isBuffer(buffer)) {
            throw new ValidationError('Invalid input: Buffer expected');
        }

        // Create multi-scale pyramid
        const pyramid = await this.createImagePyramid(buffer);
        
        // Analyze edges at each scale with enhanced error handling
        const scaleAnalyses = await Promise.allSettled(
            pyramid.map(async (level, scale) => ({
                scale: Math.pow(2, scale),
                ...(await this._safeAnalyze(level))
            }))
        );

        // Process results with fallbacks
        const validAnalyses = scaleAnalyses
            .filter(result => result.status === 'fulfilled')
            .map(result => result.value);

        if (validAnalyses.length === 0) {
            throw new ProcessingError('No valid scale analyses produced');
        }

        // Generate final results
        const edgeMap = await this.generateEdgeMap(buffer);
        const features = await this.extractEdgeFeatures(buffer, validAnalyses);

        return {
            version: '2.0',
            timestamp: Date.now(),
            multiScale: validAnalyses,
            edgeMap,
            features,
            quality: {
                completeness: validAnalyses.length / pyramid.length,
                reliability: this.assessAnalysisReliability(validAnalyses),
                confidence: this.calculateConfidenceMetrics(validAnalyses)
            }
        };

    } catch (error) {
        if (error instanceof ValidationError) {
            throw error;
        }
        throw new ProcessingError('Edge analysis failed', { error });
    } finally {
        monitor.endOperation(opId);
    }
}

    static async createImagePyramid(buffer) {
        const pyramid = [buffer];
        let currentBuffer = buffer;

        try {
            const metadata = await sharp(buffer).metadata();
            const minDimension = Math.min(metadata.width, metadata.height);
            const maxLevels = Math.floor(Math.log2(minDimension / 32)); // Minimum size 32px

            for (let i = 0; i < maxLevels; i++) {
                currentBuffer = await sharp(currentBuffer)
                    .resize(
                        Math.round(metadata.width / Math.pow(2, i + 1)),
                        Math.round(metadata.height / Math.pow(2, i + 1)),
                        {
                            kernel: sharp.kernel.lanczos3,
                            fastShrinkOnLoad: true
                        }
                    )
                    .toBuffer();
                pyramid.push(currentBuffer);
            }

            return pyramid;
        } catch (error) {
            logger.error('Image pyramid creation failed:', error);
            throw error;
        }
    }

    static async analyzeSingleScale(buffer) {
        try {
            // Enhanced noise reduction
            const smoothed = await this.applyAdaptiveSmoothing(buffer);

            // Calculate gradients with improved accuracy
            const [magnitude, direction] = await this.calculateEnhancedGradients(smoothed);

            // Advanced edge detection pipeline
            const edges = await this.detectEdges(magnitude, direction);

            // Extract comprehensive edge features
            const features = await this.extractScaleFeatures(edges, magnitude, direction);

            return {
                edges,
                magnitude,
                direction,
                features,
                quality: this.assessScaleQuality(edges, magnitude)
            };
        } catch (error) {
            logger.error('Single scale analysis failed:', error);
            throw error;
        }
    }

    static async applyAdaptiveSmoothing(buffer) {
        try {
            // Estimate noise level
            const noiseLevel = await this.estimateNoiseLevel(buffer);
            
            // Select appropriate kernel size based on noise
            const kernelSize = noiseLevel > 0.1 ? 5 : 3;
            const kernel = this.GAUSSIAN_KERNELS[kernelSize];

            // Apply smoothing with edge preservation
            return await sharp(buffer)
                .linear(kernel.flat(), kernel.flat().reduce((a, b) => a + b, 0))
                .toBuffer();
        } catch (error) {
            logger.error('Adaptive smoothing failed:', error);
            throw error;
        }
    }

    static async calculateEnhancedGradients(buffer) {
        try {
            // Calculate gradients in parallel
            const [horizontalGrad, verticalGrad] = await Promise.all([
                sharp(buffer)
                    .recomb(this.SOBEL_KERNELS.horizontal)
                    .raw()
                    .toBuffer(),
                sharp(buffer)
                    .recomb(this.SOBEL_KERNELS.vertical)
                    .raw()
                    .toBuffer()
            ]);

            // Enhanced gradient calculation with sub-pixel accuracy
            const { data: hGrad } = horizontalGrad;
            const { data: vGrad } = verticalGrad;
            const magnitude = new Float32Array(hGrad.length);
            const direction = new Float32Array(hGrad.length);

            for (let i = 0; i < hGrad.length; i++) {
                magnitude[i] = Math.sqrt(hGrad[i] * hGrad[i] + vGrad[i] * vGrad[i]);
                direction[i] = Math.atan2(vGrad[i], hGrad[i]);
            }

            return [magnitude, direction];
        } catch (error) {
            logger.error('Gradient calculation failed:', error);
            throw error;
        }
    }

    static async detectEdges(magnitude, direction) {
        try {
            // Apply non-maximum suppression
            const suppressed = await this.enhancedNonMaxSuppression(magnitude, direction);

            // Double threshold with hysteresis
            const edges = await this.adaptiveHysteresisThresholding(suppressed);

            // Post-process edges
            return this.postProcessEdges(edges);
        } catch (error) {
            logger.error('Edge detection failed:', error);
            throw error;
        }
    }

    static async enhancedNonMaxSuppression(magnitude, direction) {
        const width = Math.sqrt(magnitude.length);
        const suppressed = new Float32Array(magnitude.length);

        for (let y = 1; y < width - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                const angle = direction[idx];
                const mag = magnitude[idx];

                // Calculate interpolation factors for sub-pixel accuracy
                const normAngle = ((angle * 180 / Math.PI + 360) % 180);
                const alpha = normAngle % 45;
                const weight = alpha / 45;

                // Get interpolated neighbor values
                const [n1, n2] = this.getInterpolatedNeighbors(
                    magnitude,
                    x,
                    y,
                    width,
                    normAngle,
                    weight
                );

                suppressed[idx] = (mag >= n1 && mag >= n2) ? mag : 0;
            }
        }

        return suppressed;
    }

    static async adaptiveHysteresisThresholding(suppressed) {
        // Calculate adaptive thresholds based on image statistics
        const nonZeroValues = Array.from(suppressed).filter(v => v > 0);
        if (nonZeroValues.length === 0) return new Float32Array(suppressed.length);

        const mean = mathUtils.calculateMean(nonZeroValues);
        const std = Math.sqrt(mathUtils.calculateVariance(nonZeroValues));

        const highThreshold = mean + 0.5 * std;
        const lowThreshold = mean - 0.5 * std;

        const width = Math.sqrt(suppressed.length);
        const result = new Float32Array(suppressed.length);
        const stack = [];

        // First pass: mark strong and weak edges
        for (let i = 0; i < suppressed.length; i++) {
            if (suppressed[i] >= highThreshold) {
                result[i] = 255;
                stack.push(i);
            } else if (suppressed[i] >= lowThreshold) {
                result[i] = 128;
            }
        }

        // Second pass: trace weak edges
        const neighbors = [
            -width - 1, -width, -width + 1,
            -1, 1,
            width - 1, width, width + 1
        ];

        while (stack.length > 0) {
            const pixel = stack.pop();

            for (const offset of neighbors) {
                const neighbor = pixel + offset;
                
                if (neighbor >= 0 && neighbor < result.length &&
                    Math.abs((neighbor % width) - (pixel % width)) <= 1) {
                    
                    if (result[neighbor] === 128) {
                        result[neighbor] = 255;
                        stack.push(neighbor);
                    }
                }
            }
        }

        // Clear remaining weak edges
        for (let i = 0; i < result.length; i++) {
            if (result[i] === 128) result[i] = 0;
        }

        return result;
    }

    static async extractEdgeFeatures(buffer, scaleAnalyses) {
        // Extract comprehensive edge features
        const [
            edgeSegments,
            junctions,
            corners,
            contours
        ] = await Promise.all([
            this.extractEdgeSegments(scaleAnalyses),
            this.detectJunctionPoints(scaleAnalyses),
            this.detectCornerPoints(buffer),
            this.extractContours(scaleAnalyses)
        ]);

        return {
            segments: this.analyzeEdgeSegments(edgeSegments),
            junctions: this.analyzeJunctions(junctions),
            corners: this.analyzeCorners(corners),
            contours: this.analyzeContours(contours),
            topology: this.analyzeEdgeTopology({
                segments: edgeSegments,
                junctions,
                corners,
                contours
            }),
            hierarchy: this.buildEdgeHierarchy({
                segments: edgeSegments,
                junctions,
                contours
            })
        };
    }

    static analyzeEdgeSegments(segments) {
        return {
            count: segments.length,
            lengthDistribution: this.analyzeSegmentLengths(segments),
            orientationDistribution: this.analyzeSegmentOrientations(segments),
            curvatureAnalysis: this.analyzeSegmentCurvatures(segments),
            connectivity: this.analyzeSegmentConnectivity(segments)
        };
    }

    static analyzeJunctions(junctions) {
        return {
            count: junctions.length,
            types: this.classifyJunctionTypes(junctions),
            spatialDistribution: this.analyzeJunctionDistribution(junctions),
            connectivity: this.analyzeJunctionConnectivity(junctions)
        };
    }

    static analyzeContours(contours) {
        return {
            count: contours.length,
            shapes: this.analyzeContourShapes(contours),
            hierarchy: this.buildContourHierarchy(contours),
            relationships: this.analyzeContourRelationships(contours)
        };
    }

    static buildEdgeHierarchy(components) {
        // Build hierarchical representation of edge structure
        const hierarchy = {
            levels: this.constructHierarchicalLevels(components),
            relationships: this.analyzeHierarchicalRelationships(components),
            statistics: this.calculateHierarchyStatistics(components)
        };

        return {
            ...hierarchy,
            quality: this.assessHierarchyQuality(hierarchy)
        };
    }

    // Quality assessment methods
    static assessScaleQuality(edges, magnitude) {
        return {
            clarity: this.calculateEdgeClarity(edges, magnitude),
            continuity: this.calculateEdgeContinuity(edges),
            significance: this.calculateEdgeSignificance(edges, magnitude),
            reliability: this.calculateEdgeReliability(edges, magnitude)
        };
    }

    static assessAnalysisReliability(analyses) {
        return {
            scaleConsistency: this.calculateScaleConsistency(analyses),
            featureReliability: this.calculateFeatureReliability(analyses),
            edgeQuality: this.calculateOverallEdgeQuality(analyses)
        };
    }

    // Helper methods for noise estimation and gradient calculation
    static async estimateNoiseLevel(buffer) {
        const { data } = await sharp(buffer)
            .greyscale()
            .raw()
            .toBuffer({ resolveWithObject: true });

        const differences = [];
        const width = Math.sqrt(data.length);

        for (let y = 1; y < width - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                const neighbors = [
                    data[idx - width - 1], data[idx - width], data[idx - width + 1],
                    data[idx - 1], data[idx + 1],
                    data[idx + width - 1], data[idx + width], data[idx + width + 1]
                ];
                
                const localVariance = mathUtils.calculateVariance(neighbors);
                differences.push(Math.sqrt(localVariance));
            }
        }

        return mathUtils.calculateMean(differences) / 255;
    }

static getInterpolatedNeighbors(magnitude, x, y, width, angle, weight) {
        let n1, n2;

        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
            n1 = this.interpolate(magnitude[y * width + (x - 1)], magnitude[y * width + x], weight);
            n2 = this.interpolate(magnitude[y * width + (x + 1)], magnitude[y * width + x], weight);
        }
        else if (angle >= 22.5 && angle < 67.5) {
            n1 = this.interpolate(magnitude[(y - 1) * width + (x + 1)], magnitude[y * width + x], weight);
            n2 = this.interpolate(magnitude[(y + 1) * width + (x - 1)], magnitude[y * width + x], weight);
        }
        else if (angle >= 67.5 && angle < 112.5) {
            n1 = this.interpolate(magnitude[(y - 1) * width + x], magnitude[y * width + x], weight);
            n2 = this.interpolate(magnitude[(y + 1) * width + x], magnitude[y * width + x], weight);
        }
        else {
            n1 = this.interpolate(magnitude[(y - 1) * width + (x - 1)], magnitude[y * width + x], weight);
            n2 = this.interpolate(magnitude[(y + 1) * width + (x + 1)], magnitude[y * width + x], weight);
        }

        return [n1, n2];
    }

    static interpolate(val1, val2, weight) {
        return val1 * (1 - weight) + val2 * weight;
    }

    static postProcessEdges(edges) {
        // Remove isolated pixels
        const width = Math.sqrt(edges.length);
        const processed = new Float32Array(edges.length);

        for (let y = 1; y < width - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                if (edges[idx] > 0) {
                    let hasNeighbor = false;
                    // Check 8-connected neighborhood
                    for (let dy = -1; dy <= 1 && !hasNeighbor; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            if (dx === 0 && dy === 0) continue;
                            if (edges[(y + dy) * width + (x + dx)] > 0) {
                                hasNeighbor = true;
                                break;
                            }
                        }
                    }
                    if (hasNeighbor) {
                        processed[idx] = edges[idx];
                    }
                }
            }
        }

        // Fill small gaps
        return this.fillEdgeGaps(processed, width);
    }

    static fillEdgeGaps(edges, width) {
        const filled = new Float32Array(edges.buffer);

        for (let y = 1; y < width - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                if (edges[idx] === 0) {
                    // Check for endpoints in neighborhood
                    const endpoints = this.findNearbyEndpoints(edges, x, y, width);
                    if (endpoints.length === 2 && 
                        this.isGapFillable(endpoints[0], endpoints[1], width)) {
                        filled[idx] = 255;
                    }
                }
            }
        }

        return filled;
    }

    static findNearbyEndpoints(edges, x, y, width) {
        const endpoints = [];
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nx = x + dx;
                const ny = y + dy;
                const idx = ny * width + nx;
                
                if (edges[idx] > 0 && this.isEndpoint(edges, nx, ny, width)) {
                    endpoints.push({x: nx, y: ny});
                }
            }
        }
        return endpoints;
    }

    static isEndpoint(edges, x, y, width) {
        let neighborCount = 0;
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                if (edges[(y + dy) * width + (x + dx)] > 0) {
                    neighborCount++;
                }
            }
        }
        return neighborCount === 1;
    }

    static isGapFillable(endpoint1, endpoint2, width) {
        const dx = Math.abs(endpoint1.x - endpoint2.x);
        const dy = Math.abs(endpoint1.y - endpoint2.y);
        return Math.sqrt(dx * dx + dy * dy) <= this.MAX_GAP_LENGTH;
    }

    static calculateEdgeClarity(edges, magnitude) {
        let totalClarity = 0;
        let edgeCount = 0;

        for (let i = 0; i < edges.length; i++) {
            if (edges[i] > 0) {
                totalClarity += magnitude[i];
                edgeCount++;
            }
        }

        return edgeCount > 0 ? totalClarity / (edgeCount * 255) : 0;
    }

    static calculateEdgeContinuity(edges) {
        const width = Math.sqrt(edges.length);
        let totalDiscontinuities = 0;
        let edgePixels = 0;

        for (let y = 1; y < width - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                if (edges[idx] > 0) {
                    edgePixels++;
                    let hasDiscontinuity = true;
                    // Check 8-connected neighborhood
                    for (let dy = -1; dy <= 1 && hasDiscontinuity; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            if (dx === 0 && dy === 0) continue;
                            if (edges[(y + dy) * width + (x + dx)] > 0) {
                                hasDiscontinuity = false;
                                break;
                            }
                        }
                    }
                    if (hasDiscontinuity) totalDiscontinuities++;
                }
            }
        }

        return edgePixels > 0 ? 1 - (totalDiscontinuities / edgePixels) : 0;
    }

    static calculateEdgeSignificance(edges, magnitude) {
        const significantThreshold = mathUtils.calculateMean(magnitude) + 
            Math.sqrt(mathUtils.calculateVariance(magnitude));

        let significantEdges = 0;
        let totalEdges = 0;

        for (let i = 0; i < edges.length; i++) {
            if (edges[i] > 0) {
                totalEdges++;
                if (magnitude[i] >= significantThreshold) {
                    significantEdges++;
                }
            }
        }

        return totalEdges > 0 ? significantEdges / totalEdges : 0;
    }

    static getDefaultAnalysis() {
        return {
            multiScale: [],
            dominant: {
                orientation: 0,
                magnitude: 0,
                density: 0
            },
            edgeMap: new Float32Array(0),
            statistics: this.getDefaultStatistics(),
            features: this.getDefaultFeatures(),
            quality: {
                completeness: 0,
                reliability: 0,
                confidence: 0
            }
        };
    }
}

module.exports = EdgeAnalysis;
